
import socket, os
import numpy as np
from tensorflow.keras import layers as KL
from tqdm import tqdm
import glob, copy
import scipy
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K

from freesurfer import deeplearn as fsd
import freesurfer as fs

import neurite as ne
import neurite_sandbox as nes
import voxelmorph as vxm
import voxelmorph_sandbox as vxms
from pathlib import Path
from tensorflow.keras.utils import to_categorical
import surfa as sf
import generators as gens

import layer_dict as ld
import pdb as gdb

mse_wt = 1

vscale = 2
vscale = 1

dofit = False
dofit = True

# whether or not to train the synthseg net from scratch with the subnets
combined_training = True
combined_training = False

# whether or not to concat the (extracted) aseg features into the subnets
concat_aseg = False
concat_aseg = True

# have subnets only output the labels that ever occur in their location or all labels
use_lab2ind = True
use_lab2ind = False

# put losses on the individual subnets or not
use_subloss = False
use_subloss = True

# which optimizer to use
which_opt = 'sgd'
which_opt = 'adam'

# which subnet loss to use (if using a subnet loss)
subloss = 'mse'
subloss = 'dice'

# whether to use the insertion code on the subnet outputs (you always should, wasn't sure at first)
use_insertion = False
use_insertion = True

# whether to freeze the aseg weights when training the subnets
train_aseg = False
train_aseg = True

which_loss = 'both'
which_loss = 'mse'
which_loss = 'cce'
which_loss = 'dice'

same_contrast=False
same_contrast=True

# add synthetic outputs shapes to the images or not
oshapes = False
oshapes = True

# perform linear fitting on inputs to softmax to initialize things
fit_lin = True
fit_lin = False


# do affine augmentation or not
doaff = False
doaff = True


model_dir = 'models'
gpuid = -1
host = socket.gethostname()
from neurite_sandbox.tf.utils.utils import plot_fit_callback as pfc


print(f'host name {socket.gethostname()}')

ngpus = 1 if os.getenv('NGPUS') is None else int(os.getenv('NGPUS'))

print(f'using {ngpus} gpus and tf version {tf.__version__}')
if ngpus > 1:
    model_device = '/gpu:0'
    synth_device = '/gpu:1'
    synth_gpu = 1
    dev_str = "0, 1"
else:
    model_device = '/gpu:0'
    synth_device = model_device
    synth_gpu = 0
    dev_str = "0"

if not dofit and host == 'serena.nmr.mgh.harvard.edu':
    dev_str = '/cpu:0'
    print(f'setting dev_str to {dev_str}')

os.environ["CUDA_VISIBLE_DEVICES"] = dev_str

print(f'model_device {model_device}, synth_device {synth_device}, dev_str {dev_str}')

print(f'physical GPU # is {os.getenv("SLURM_STEP_GPUS")}')
ret = ne.utils.setup_device(dev_str)
policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')

print(f'dofit {dofit}, doaff {doaff}, fit_lin {fit_lin}, oshapes {oshapes}, subnet loss {use_subloss}, use_lab2ind {use_lab2ind}, combined_training {combined_training}, insertion {use_insertion}, concat aseg {concat_aseg}, subloss {subloss}')


batch_size = 1

odir = '/autofs/vast/braindev/braindev/OASIS/OASIS1/synth-high-res/recon_subject'
subjects = [f for f in Path(odir).iterdir() if 'OASIS_OAS1_0' in str(f)]
seg_files = [f/f'mri/aparc+aseg.mgz' for f in tqdm(subjects)]
seg_files = [f/f'mri/aseg.mgz' for f in tqdm(subjects)]

if dofit:
    print(f'TRAINING model with loss {which_loss}')
else:
    print(f'loading model trained with {which_loss} loss')

target_shape = (192,)*3
inshape = target_shape


# number of subnets and the size of the input patch to each one
patch_size = 32
num_subnets = 16

# add some padding to the subnets so that there is always some context for each voxel
big_patch_size = patch_size + 2 * pad_size
phalf = patch_size // 2
bphalf = big_patch_size // 2

print(f'patch_size = {patch_size}, num_subnets = {num_subnets}, pad_size = {pad_size}')

ntest = 25
crop = -1 if dofit else ntest
crop = -1

lut = fs.lookups.default()
# lut = fs.lookups.LookupTable.read(os.path.join(odir, 'seg35_labels.txt'))
# lut = fs.lookups.LookupTable.read(os.path.join(odir, f'seg{nb_labels}_labels.txt'))

# csf = lut.search('CSF')[0]
lesion_label_orig = lut.search('Left-Lesion')
if len(lesion_label_orig) > 0:
    lesion_label_orig = lesion_label_orig[0]
else:   # not in the lut - add a new one
    lesion_label_orig = 77
    lut.add(lesion_label_orig, 'Lesion', color=[240,240,240])

if 'inited' not in locals() and 'inited' not in globals():
    inited = False

warp_max = 2
if not inited:
    mri_segs_orig = [fs.Volume.read(str(fname)) for fname in tqdm(seg_files[:crop])]
    # mri_segs = [sf.load_volume(str(fname)) for fname in tqdm(seg_files[:crop])]
    if vscale > 1:
        print(f'downsampling by {vscale}')
        mri_segs = [mri.reslice(vscale) for mri in tqdm(mri_segs_orig)]
    else:
        print(f'cropping to {target_shape}')
        mri_segs = [mri.fit_to_shape(target_shape, center='bbox') for mri in tqdm(mri_segs_orig)]
        # mri_segs = [mri.reshape(target_shape, center='bbox') for mri in tqdm(mri_segs)]

    np_segs_orig = [mri.data for mri in mri_segs]
    fname = 'oasis_labels.npy'
    if os.path.exists(fname):
        print(f'loading unique labels in {len(mri_segs_orig)} from {fname}')
        labels_orig = np.load(fname)
    else:
        print(f'finding unique labels in {len(mri_segs_orig)} datasets...')
        labels_orig = np.unique(np.array(np_segs_orig))
        np.save(fname, labels_orig)

    # mapping = fs.lookups.tissue_type_recoder_no_skull(include_lesions=use_lesions)
    mapping = fs.lookups.nonlateral_aseg_recoder()
    target_lut = mapping.target_lut
    lut_name = 'nonlat.txt'
    mri_segs_recoded = [fs.label.recode(mri, mapping) for mri in mri_segs]  
    lesion_label = target_lut.search('Left-Lesion')[0]
    np_segs = [vol.data for vol in mri_segs_recoded]
    labels_in = np.unique(np.array(np_segs)).astype(int)
    if lesion_label not in labels_in:
        l = list(labels_in)
        l.append(lesion_label)
        labels_in = np.array(l)
    nlabels_small = len(labels_in)
    label_map = {}
    keys = mapping.mapping.keys()
    lab_to_ind = np.zeros((labels_orig.max()+1,), dtype=int)
    for label in labels_orig:
        if label not in keys:
            output_label = 0
        else:
            output_label = mapping.mapping[label]
        label_map[label] = output_label
        lab_to_ind[label] = output_label


    import generators as gens
    vxm_model = gens.read_vxm_model(inshape)
    vxm_smooth_wt = np.zeros((1, 1))
    vxm_smooth_wt[0,0] = .3   # warp regularization hyper parameter
    
    gen_model = gens.create_gen_model(np_segs, oshapes, synth_device, nlabels_small, labels_in, inshape, warp_max)

    mapping = fs.lookups.nonlateral_aseg_recoder()
    target_lut = mapping.target_lut
    lut_name = 'nonlat.txt'
    adir_2d = '/autofs/cluster/vxmdata1/FS_Slim/proc/cleaned/Buckner39'
    adir = '/autofs/cluster/freesurfer/subjects/atlases/aseg_atlas'
    mname = 'seg_edited.mgz'
    vname = 'norm.mgz'
    sfile = os.path.join(adir, 'scripts', 'subjects.txt')
    with open(sfile, 'r') as f:
        subjects = f.read().split('\n')[0:-1]

    mri_man_segs = []  # manual segs
    mri_norms = []  # mri vols
    mri_norms_orig = []
    mri_man_segs_orig = []
    for s in tqdm(subjects):
        mri_seg_orig = fs.Volume.read(os.path.join(adir, s, 'mri', mname))
        mri_man_segs_orig.append(mri_seg_orig)
        mri_seg = mri_seg_orig.fit_to_shape(target_shape, center='bbox')
        mri_man_segs.append(mri_seg)
        mri_norm_orig = fs.Volume.read(os.path.join(adir, s, 'mri', vname))
        mri_norm = mri_norm_orig.resample_like(mri_seg)
        mri_norms.append(mri_norm)
        mri_norms_orig.append(mri_norm_orig)

    mri_man_segs_recoded = [fs.label.recode(mri, mapping) for mri in tqdm(mri_man_segs)]

    mri_seg_atlas = fs.Volume.read("aseg_atlas.mgz")
    hard_seg = np.argmax(mri_seg_atlas.data, axis=-1)
    mri_hard_seg = mri_seg_atlas.copy(hard_seg).fit_to_shape(target_shape, center='bbox')
    mri_norm_atlas = fs.Volume.read("norm_atlas.mgz").resample_like(mri_hard_seg)
    mri_seg_atlas = mri_seg_atlas.resample_like(mri_hard_seg)
    norm_atlas = (mri_norm_atlas.data / mri_norm_atlas.data.max())[np.newaxis, ..., np.newaxis]

    f = 128
    conf = {
        'def.enc_nf': [f] * 4,
        'def.dec_nf': [f] * 4,
        'def.add_nf': [f] * 4,
        'def.hyp_den': [32] * 4,
    }

    vxm_model = vxms.networks.VxmJointAverage(in_shape=inshape, **conf)
    vxm_model.load_weights(os.path.join('models_from_Malte', f'VxmJointAverage{f}.h5'))
    #aseg_model = tf.keras.models.load_model('aseg.h5', custom_objects=ld.layer_dict)
    aseg_train_fscale = 1.1
    if 1:
        aseg_train_fname = f'aseg.fscale.{aseg_train_fscale}.h5'
        print(f'loading aseg model {aseg_train_fname}')
        aseg_model = tf.keras.models.load_model(aseg_train_fname, 
                                                custom_objects=ld.layer_dict)
    else:
        aseg_model = tf.keras.models.load_model('aseg.h5', custom_objects=ld.layer_dict)

    l = np.zeros((1, 1))
    l[0,0] = .3   # warp regularization hyper parameter

    # psize = (norm_atlas.shape[1] - mri_norms[0].shape[0]) // 2
    # pad = ((0,0), (psize,psize), (psize, psize), (psize, psize), (0, 0))
    # lfunc = ne.losses.Dice(nb_labels=nlabels_small, weights=None, check_input_limits=False).mean_loss
    lfunc = nes.losses.DiceNonzero(nlabels_small, weights=None, check_input_limits=False).loss
    read_cached = False
    read_cached = True
    new_cache = True  # fscale specific
    if not read_cached:
        dice_list = []
        elist = []
        elist_in_atlas = []
        alist_in_atlas = []
        nlist_in_atlas = []
        mlist_in_atlas = []
        for sno, s in enumerate(tqdm(subjects)):
            mseg_onehot = np.eye(nlabels_small)[mri_man_segs_recoded[sno].data]
            norm = (mri_norms[sno].data / mri_norms[sno].data.max())[np.newaxis, ..., np.newaxis]
            pred = aseg_model.predict(norm)
            transform = vxm_model.predict([l, norm, norm_atlas])
            # transform = transform[:, psize:-psize, psize:-psize, psize:-psize, :]
            dice = lfunc(tf.convert_to_tensor(mseg_onehot[np.newaxis], tf.float32), 
                         tf.convert_to_tensor(pred, tf.float32))
            dice_list.append(dice.numpy())
            ev = (mseg_onehot - pred[0])
            evol = (ev**2).sum(axis=-1)
            elist.append(evol)
            evol_in_atlas = vxm.layers.SpatialTransformer(interp_method='linear', fill_value=0)([evol[np.newaxis, ..., np.newaxis], transform])
            norm_in_atlas = vxm.layers.SpatialTransformer(interp_method='linear', fill_value=0)([norm, transform])
            aseg_in_atlas = vxm.layers.SpatialTransformer(interp_method='linear', fill_value=0)([pred[0][np.newaxis, ...], transform])
            mseg_in_atlas = vxm.layers.SpatialTransformer(interp_method='linear', fill_value=0)([mseg_onehot[np.newaxis, ...], transform])
            elist_in_atlas.append(evol_in_atlas.numpy().squeeze())
            nlist_in_atlas.append(norm_in_atlas.numpy().squeeze())
            #alist_in_atlas.append(np.argmax(aseg_in_atlas.numpy(), axis=-1).squeeze())
            #mlist_in_atlas.append(np.argmax(mseg_in_atlas.numpy(), axis=-1).squeeze())
            alist_in_atlas.append(aseg_in_atlas.numpy().squeeze())
            mlist_in_atlas.append(mseg_in_atlas.numpy().squeeze())

        np.save(f'elist_in_atlas.fscale.{aseg_train_fscale}.npy', elist_in_atlas)
        np.save(f'nlist_in_atlas.fscale.{aseg_train_fscale}.npy', nlist_in_atlas)
        np.save(f'alist_in_atlas.fscale.{aseg_train_fscale}.npy', alist_in_atlas)
        np.save(f'mlist_in_atlas.fscale.{aseg_train_fscale}.npy', mlist_in_atlas)

        print('computing index occurence volumes')
        aseg_inds = np.argmax(np.array(alist_in_atlas), axis=-1)
        mseg_inds = np.argmax(np.array(mlist_in_atlas), axis=-1)
        max_inds = nlabels_small
        mseg_ind_vol = np.zeros(target_shape + (max_inds,))
        aseg_ind_vol = np.zeros(target_shape + (max_inds,))
        mseg_num_inds = np.zeros(target_shape)
        aseg_num_inds = np.zeros(target_shape)
        for x in tqdm(range(aseg_inds.shape[1])):
            for y in range(aseg_inds.shape[2]):
                for z in range(aseg_inds.shape[3]):
                    ind_list = -1 * np.ones((max_inds,))
                    u = np.unique(mseg_inds[:, x, y, z])
                    for lno, l in enumerate(u):
                        if lno >= max_inds:
                            break
                            mseg_ind_vol[x, y, z, lno] = l
                
                    mseg_num_inds[x, y, z] = len(u)
                    ind_list = -1 * np.ones((max_inds,))
                    u = np.unique(aseg_inds[:, x, y, z])
                    for lno, l in enumerate(u):
                        aseg_ind_vol[x, y, z, lno] = l

                    aseg_num_inds[x, y, z] = len(u)

        inds_trivial = np.nonzero(mseg_num_inds < 2)
        evol_avg = np.array(elist_in_atlas).mean(axis=0)
        evol_avg[inds_trivial] = 0
        nvol_avg = np.array(nlist_in_atlas).mean(axis=0)
        low_val = 0 if num_subnets == 20 else -1000
        nbhd_img_orig = scipy.ndimage.convolve(evol_avg, np.ones((patch_size,)*3)/(patch_size**3), 
                                               mode='constant',  cval=low_val)
        np.save(f'nbhd_img.fscale.{aseg_train_fscale}.npy', nbhd_img_orig)
        np.save(f'evol_avg.fscale.{aseg_train_fscale}.npy', evol_avg)
        np.save(f'nvol_avg.fscale.{aseg_train_fscale}.npy', nvol_avg)
    else:   # not dofit
        if new_cache:
            print(f'reading cached volumes scale {aseg_train_fscale}')
            nbhd_img_orig = np.load('nbhd_img.npy', allow_pickle=True)
            elist_in_atlas = np.load(f'elist_in_atlas.fscale.{aseg_train_fscale}.npy', allow_pickle=True)
            nlist_in_atlas = np.load(f'nlist_in_atlas.fscale.{aseg_train_fscale}.npy', allow_pickle=True) 
            alist_in_atlas = np.load(f'alist_in_atlas.fscale.{aseg_train_fscale}.npy', allow_pickle=True)
            mlist_in_atlas = np.load(f'mlist_in_atlas.fscale.{aseg_train_fscale}.npy', allow_pickle=True)
            evol_avg = np.load(f'evol_avg.fscale.{aseg_train_fscale}.npy', allow_pickle=True)
            nvol_avg = np.load(f'nvol_avg.fscale.{aseg_train_fscale}.npy', allow_pickle=True)
        else:
            print('reading cached volumes')
            nbhd_img_orig = np.load('nbhd_img.npy', allow_pickle=True)
            elist_in_atlas = np.load('elist_in_atlas.npy', allow_pickle=True)
            nlist_in_atlas = np.load('nlist_in_atlas.npy', allow_pickle=True)
            alist_in_atlas = np.load('alist_in_atlas.npy', allow_pickle=True)
            mlist_in_atlas = np.load('mlist_in_atlas.npy', allow_pickle=True)
            evol_avg = np.load('evol_avg.npy', allow_pickle=True)
            nvol_avg = np.load('nvol_avg.npy', allow_pickle=True)

    inited = True


nbhd_img = nbhd_img_orig.copy()
mseg_avg = np.argmax(np.array(mlist_in_atlas).mean(axis=0), axis=-1)
aseg_avg = np.argmax(np.array(alist_in_atlas).mean(axis=0), axis=-1)
    
mlist_ind = np.argmax(np.array(mlist_in_atlas), axis=-1)
alist_ind = np.argmax(np.array(alist_in_atlas), axis=-1)

if 0:
    fv = fs.Freeview()
    fv.vol(mri_segs[0].copy(nvol_avg), name='norm avg')
    fv.vol(mri_segs[0].copy(mseg_avg), name='mseg', opts=':colormap=lut:lut=nonlat')
    fv.vol(mri_segs[0].copy(aseg_avg), name='aseg', opts=':colormap=lut:lut=nonlat')
    if 0:
        nvols = 3
        mvols = np.argmax(np.array(mlist_in_atlas[0:nvols]), axis=-1)
        avols = np.argmax(np.array(alist_in_atlas[0:nvols]), axis=-1)
        
        fv.vol(mri_segs[0].copy(np.transpose(np.array(nlist_in_atlas[0:nvols]), (1, 2, 3, 0))), name='norm vols')
        fv.vol(mri_segs[0].copy(np.transpose(mvols, (1, 2, 3, 0))), name='man segs', opts=':colormap=lut:lut=nonlat')
        fv.vol(mri_segs[0].copy(np.transpose(avols, (1, 2, 3, 0))), name='synth segs', opts=':colormap=lut:lut=nonlat')
    fv.vol(mri_segs[0].copy(nbhd_img), name='err avg', opts=':colormap=heat:heatscale=.25,.5')
    fv.vol(mri_segs[0].copy(mseg_num_inds), name='mseg ninds', opts=':colormap=heat:heatscale=2,5')
    fv.vol(mri_segs[0].copy(aseg_num_inds), name='mseg ninds', opts=':colormap=heat:heatscale=2,5')
    fv.show(opts=f'-slice {patch_center[0]} {patch_center[1]} {patch_center[2]}', verbose=True)

if dofit:    #  run training
    patch_labels = []
    patch_centers = []
    for netno in range(num_subnets):
        max_ind = np.argmax(nbhd_img[bphalf:-bphalf, bphalf:-bphalf, bphalf:-bphalf])
        patch_center = np.unravel_index(max_ind, nbhd_img[bphalf:-bphalf, bphalf:-bphalf, bphalf:-bphalf].shape) + np.array([bphalf, bphalf, bphalf])
        # should use bphalf here
        x0 = patch_center[0]-patch_size//2
        x1 = x0 + patch_size 
        y0 = patch_center[1]-patch_size//2
        y1 = y0 + patch_size 
        z0 = patch_center[2]-patch_size//2
        z1 = z0 + patch_size
        labels_mseg = np.unique(mlist_ind[:, x0:x1, y0:y1, z0:z1])
        labels_aseg = np.unique(alist_ind[:, x0:x1, y0:y1, z0:z1])
        print(f'patch at {patch_center}: labels {labels_mseg}')
        patch_centers.append(patch_center)
        bx0 = patch_center[0]-big_patch_size//2
        bx1 = x0 + big_patch_size 
        by0 = patch_center[1]-big_patch_size//2
        by1 = y0 + big_patch_size 
        bz0 = patch_center[2]-big_patch_size//2
        bz1 = z0 + big_patch_size
        nbhd_img[bx0:bx1, by0:by1, bz0:bz1] = -1000  # prevent overlap of subsequent patches
        patch_labels.append(labels_mseg)

    print('saving patch centers and labels...')
    np.save(f'patch_labels.{patch_size}.{num_subnets}.{pad_size}.npy', patch_labels)
    np.save(f'patch_centers.{patch_size}.{num_subnets}.{pad_size}.npy', patch_centers)
else:
    print('loading patch centers and labels...')
    patch_labels = np.load(f'patch_labels.{patch_size}.{num_subnets}.{pad_size}.npy', allow_pickle=True)
    patch_centers = np.load(f'patch_centers.{patch_size}.{num_subnets}.{pad_size}.npy', allow_pickle=True)


if combined_training:   # train synthseg net and subnets from scratch
    nfeats = 64
    unet_nf = []
    nb_conv_per_level = 2
    fscale = 1
    fscale = 1.1
    nb_levels = int(np.log2(inshape[0]))-(1)   # 4,4,4 is lowest level

    for level in range(nb_levels):
        filters_in_this_level = []
        for layer in range(nb_conv_per_level):
            filters_in_this_level.append(int(fscale**level*nfeats))
        
        unet_nf.append(filters_in_this_level)

    model_lin = ne.models.unet(unet_nf, inshape+(1,), None, 3, nlabels_small, feat_mult=None, final_pred_activation='linear')
    softmax_out = KL.Softmax(name='seg')(model_lin.outputs[0])
    aseg_model = tf.keras.Model(model_lin.inputs, [softmax_out])
else:
    aseg_train_fname = f'aseg.fscale.{aseg_train_fscale}.h5'
    print(f'loading aseg model {aseg_train_fname}')
    aseg_model = tf.keras.models.load_model(aseg_train_fname, custom_objects=ld.layer_dict)

# find last feature layer of the aseg unet
for lno in range(len(aseg_model.layers)-1, 0, -1):
    aseg_layer = aseg_model.layers[lno]
    if aseg_layer.name.startswith("unet_conv_uparm"):
        break

nfeats = aseg_layer.output[0].get_shape().as_list()[-1]


gen_model = gens.create_gen_model(np_segs, oshapes, synth_device, nlabels_small, labels_in, inshape)

# generate a  synth image
# inputs are synth images
# outputs are label maps
# build lists for unet architecture
nb_levels = int(np.log2(patch_size))-1
nb_levels = int(np.log2(big_patch_size))-1
nb_conv_per_level = 2
unet_scale = 1
    
unet_nf = []
fscale = 1.1
fscale = 1


for level in range(nb_levels):
    filters_in_this_level = []
    for layer in range(nb_conv_per_level):
        filters_in_this_level.append(int(fscale**level*nfeats))
        
    unet_nf.append(filters_in_this_level)


tf.compat.v1.enable_eager_execution()

lr = 1e-5
lr = 1e-4
name = f'subnets.outside.unet_nf.{nfeats}.warp_max.{warp_max}.oshapes.{oshapes}.num_subnets.{num_subnets}.psize.{patch_size}.pad.{pad_size}.lab2ind.{use_lab2ind}.lr.{lr}.subloss.{use_subloss}.insertion.{use_insertion}.combined_training.{combined_training}.train_aseg.{train_aseg}'
if not concat_aseg:
    name += f'.concat_aseg.{concat_aseg}'
if use_subloss and subloss != 'dice':
    name += f'.subloss.{subloss}'
if which_opt != 'adam':
    name += f'.which_opt.{which_opt}'

class val_loss:
    def __init__(self, lfunc, label_mask):
        self.lfunc = lfunc
        self.label_mask = label_mask

    def loss(yt, yp):
        lvals = self.lfunc(yt, yp)
        lval = tf.reduce_sum(lvals * self.label_mask) / tf.reduce_sum(self.label_mask)
        return lval

label_weights = np.ones((1,nlabels_small,))
# label_weights[0,-1] = .01  # downweight lesion class
lfunc = ne.losses.Dice(nb_labels=nlabels_small, weights=None, check_input_limits=False).mean_loss
lfunc = nes.losses.DiceNonzero(nlabels_small, weights=None, check_input_limits=False).loss
if use_subloss:
    if subloss == 'dice':
        thresh = -.2*3
    else:
        thresh = 8
else:
        thresh = -.2*2
    

cooldown = 25
patience = 600

write_cb_lin = nes.callbacks.WriteHist(name+'.lin.txt')

losses = [lfunc]
loss_weights = [1]
if use_subloss:
    if subloss == 'mse':
        lfunc_subnet = lambda a, b: mse_wt * tf.keras.losses.MSE(a, b)
    else:
        lfunc_subnet = lfunc
    #lfunc_mse = tf.keras.losses.mse
    losses += [lfunc_subnet]
    loss_weights += [1]


unet_device = model_device if (fit_lin or dofit) else synth_device
with tf.device(unet_device):     # add the subnets to the big unet
    aseg_shape = aseg_layer.output.get_shape().as_list()[1:]
    aseg_linear_out = aseg_model.layers[-3].output  # tensor right before outputs compressed to nlabels
    subnet_outputs_to_add = [aseg_linear_out]  # add subnet outputs to this tensor for final output
    subnet_outputs = []  # list of subnet outputs each nlabels_small long (only for using subloss)
    subnet_patches = []  # spatial location of the subnet patches
    pre_lab2ind = []

    for subnet_no in range(num_subnets):
        bx0 = patch_centers[subnet_no][0]-big_patch_size//2
        bx1 = bx0 + big_patch_size 
        by0 = patch_centers[subnet_no][1]-big_patch_size//2
        by1 = by0 + big_patch_size 
        bz0 = patch_centers[subnet_no][2]-big_patch_size//2
        bz1 = bz0 + big_patch_size 
        # only inner part without padding
        x0 = patch_centers[subnet_no][0]-patch_size//2
        x1 = x0 + patch_size 
        y0 = patch_centers[subnet_no][1]-patch_size//2
        y1 = y0 + patch_size 
        z0 = patch_centers[subnet_no][2]-patch_size//2
        z1 = z0 + patch_size 
        # labels_mseg = np.unique(mlist_ind[:, bx0:bx1, by0:by1, bz0:bz1])
        labels_mseg = np.unique(mlist_ind[:, x0:x1, y0:y1, z0:z1])
        noutputs = len(labels_mseg) if use_lab2ind else nlabels_small

        # extract a patch of (1) the data on the uparm of the big net and (2) the input volume
        aseg_patch = nes.layers.ExtractPatch(((bx0, bx1), (by0, by1), (bz0, bz1)), 
                                             name=f'aseg_patch{subnet_no}')(aseg_layer.output)
        patch_input = nes.layers.ExtractPatch(((bx0, bx1), (by0, by1), (bz0, bz1)),
                                              name=f'subnet_input{subnet_no}')(aseg_model.inputs[0])
        nf = aseg_patch.get_shape().as_list()[-1]
        subnet_lin = ne.models.unet(unet_nf, (big_patch_size,)*3+(1,), None, 3, noutputs, feat_mult=None, final_pred_activation='linear', name=f'subnet{subnet_no}')

        if use_lab2ind:
            # concat aseg_model info to second-to-last layer of subnet
            if concat_aseg:
                tmp_model = tf.keras.Model(subnet_lin.inputs, subnet_lin.layers[-3].output)
                tmp_out = tmp_model(patch_input)
                unet_concat = KL.Concatenate(name=f'subnet_in{subnet_no}', axis=-1)([tmp_out, aseg_patch])
                Conv = getattr(KL, 'Conv%dD' % 3)
                subnet_out = Conv(noutputs, 3, strides=1, padding='same')(unet_concat)
            else:
                subnet_out = subnet_lin(patch_input)
            
            pre_lab2ind.append(subnet_out)
            unet_out = nes.layers.IndexToLabel(labels_mseg, nlabels_small, name=f'IndToLab{subnet_no}')(subnet_out)
        else:  # what about concatting in the non lab2ind case????
            unet_out = subnet_lin(patch_input)

        # crop out the beginning and ending pad regions so output is just central patch_size
        p0 = pad_size
        p1 = pad_size+patch_size
        patch_output = nes.layers.ExtractPatch(((p0, p1), (p0, p1), (p0, p1)),
                                               name=f'subnet_output{subnet_no}')(unet_out)

        if use_subloss:
            if subloss == 'dice':
                subnet_softmax = KL.Softmax(name=f'subnet{subnet_no}_softmax')(patch_output)
                subnet_outputs.append(subnet_softmax[:, tf.newaxis, ...])
            else:   # just include the linear output for mse loss
                subnet_outputs.append(patch_output[:, tf.newaxis, ...])

            subnet_patches.append(((x0, x1), (y0, y1), (z0, z1)))

        if subnet_no == 0 or not use_insertion:
            padding = ((x0, aseg_shape[0]-x1), (y0, aseg_shape[1]-y1), (z0, aseg_shape[2]-z1))
            padded_unet_output = nes.layers.Pad(padding=padding, mode='constant', name=f'pad_subnet{subnet_no}')(
                patch_output)
            if not use_insertion:
                subnet_outputs_to_add.append(padded_unet_output)
            else:   # initialize one big tensor with patch outputs
                big_patches_output = padded_unet_output
        else:  # just insert this patch (since they don't overlap don't need to add)
            offset = ((x0, y0, z0, 0))
            big_patches_output = nes.layers.InsertPatch(big_patches_output, offset, 
                                                         name=f'patch_insert{subnet_no}')([
                                                             patch_output, big_patches_output])
            

    if not use_insertion:
        summed_aseg_outputs = KL.Add(name='patch_plus_unet')(subnet_outputs_to_add)
    else:
        summed_patch_outputs = KL.Lambda(lambda x: x, name='summed_patch_outputs')(big_patches_output)
        summed_aseg_outputs = KL.Add(name='patch_plus_unet')([summed_patch_outputs, aseg_linear_out])

    #Conv = getattr(KL, 'Conv%dD' % 3)
    #linear_out = Conv(nlabels_small, 3, strides=1, padding='same')(summed_aseg_outputs)

    model_lin = tf.keras.Model(aseg_model.inputs, [summed_aseg_outputs])
    softmax_out = KL.Softmax(name='seg')(summed_aseg_outputs)
    outputs = [softmax_out]
    if use_subloss:
        subnet_out = KL.Concatenate(name='subloss', axis=1)(subnet_outputs)
        outputs += [subnet_out]
    model = tf.keras.Model(aseg_model.inputs, outputs)


# create the training (synth) and validation (on real data) generators
with tf.device(synth_device):
    val_size = 20
    gen = gens.synth_gen(mri_segs_recoded, gen_model, vxm_model, norm_atlas, 
                         None, labels_in, 
                         batch_size=batch_size, 
                         subnet_patches=subnet_patches if use_subloss else None,
                         use_log_for_subnet=subloss == 'mse',
                         use_rand=True, gpuid=synth_gpu, debug=False, add_outside=oshapes)
    vgen = gens.real_gen(mri_man_segs_recoded, mri_norms, vxm_model, norm_atlas, 
                         None, labels_in, 
                         batch_size=batch_size, use_rand=True,
                         use_log_for_subnet=subloss == 'mse',
                         subnet_patches=subnet_patches if use_subloss else None,
                         gpuid=synth_gpu, debug=False, add_outside=oshapes)



# set this to nonzero if restarting training from an interrupted run
initial_epoch = 1
initial_epoch = 0

if not use_subloss:
    key_replacements = {
        'loss' : 'seg_loss',
        'val_loss' : 'val_seg_loss'
    }
else:
    key_replacements = None


opt = keras.optimizers.Adam(learning_rate=lr) if which_opt == 'adam' else keras.optimizers.SGD(learning_rate=lr)
nes.utils.check_and_compile(model, gen, optimizer=opt, 
                            loss=losses, loss_weights=loss_weights, check_layers=False, run_eagerly=True)
write_cb = nes.callbacks.WriteHist(name+'.txt', mode='w' if initial_epoch == 0 else 'a', key_replacements=key_replacements)
if initial_epoch > 0:
    initial_epoch = write_cb.start_epoch
    print(f'loading old model and restarting from epoch {initial_epoch}')
    # model = tf.keras.models.load_model(name+'.checkpoint.h5', custom_objects=ld.layer_dict)

mc_cb = tf.keras.callbacks.ModelCheckpoint(name+'.checkpoint.h5', save_best_only=True, include_optimizer=False)
lr_cb = nes.tf.callbacks.ReduceLRWithModelCheckpointAndRecovery(name+'.tf', monitor='loss',
                                                                verbose=2, cooldown=cooldown, 
                                                                recovery_decrease_factor=1,
                                                                factor=.8, patience=patience, 
                                                                thresh_increase_factor=1.2,
                                                                thresh=thresh, 
                                                                include_optimizer=False,
                                                                save_weights_only=True, 
                                                                burn_in=50,
                                                                min_lr=1e-7,
#                                                                restart=2 if initial_epoch > 0 else 0,
                                                                restart=initial_epoch > 0,
                                                                nloss=5)
#callbacks = [lr_cb, write_cb, ne.callbacks.LRLog(), mc_cb]
callbacks = [lr_cb, write_cb, ne.callbacks.LRLog()]

checkpoint = tf.train.Checkpoint(model=model,optim=model.optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, f'{name}.saved_model', max_to_keep = 5)
cp = checkpoint_manager.restore_or_initialize()
if initial_epoch > 0:
    status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
    
with tf.device(model_device):
    if not train_aseg:
        aseg_model.trainable = False
        
    print(f"{'saving' if dofit else 'loading'} fit results to {lr_cb.fname} and hist to {write_cb.fname}")
    if dofit:
        fhist = model.fit(gen, epochs=int(10000), steps_per_epoch=50, 
                          initial_epoch=initial_epoch, callbacks=callbacks, 
                          validation_data=vgen, validation_steps=5)
    else:   # load a previous model and do inference with it
        # model = tf.keras.models.load_model(name+'.checkpoint.h5', custom_objects=ld.layer_dict)
        model.load_weights(lr_cb.fname)
        sum_layer = model.get_layer('patch_plus_unet')
        padded_outputs = sum_layer.input[1:]
        patch_inputs = []
        subnet_type = type(subnet_lin)
        patch_outputs = []
        patch_unets = []
        patch_layers = []
        for layer in model.layers:
            if layer.name.startswith('subnet_input'):
                patch_inputs.append(layer.output)

            if type(layer) == subnet_type:
                patch_unets.append(layer)

            if layer.name.startswith('pad_subnet'):
                patch_outputs.append(layer.input)
                patch_layers.append(layer)

        if use_insertion:
            patch_layer = model.get_layer('summed_patch_outputs')
            patch_outputs = [patch_layer.input[0]]

        model2 = tf.keras.Model(model.inputs, model.outputs + padded_outputs)
        if len(subnet_outputs_to_add) > 2:
            summed_patch_outputs = KL.Add(name='patch_sum')(subnet_outputs_to_add[1:])
        elif use_insertion:
            summed_patch_outputs = big_patches_output
        else:
            summed_patch_outputs = subnet_outputs_to_add[-1]  # only 1 patch

        pmodel = tf.keras.Model(model.inputs, patch_inputs + [summed_patch_outputs])
        if 0:
            inb, outb = next(gen)
            
            p = pmodel.predict(inb)
            pin = np.array(p[0:num_subnets]).squeeze()
            pout = np.argmax(np.array(p[num_subnets:]).squeeze(), axis=-1)
            fv = fs.Freeview(swap_batch_dim=True)
            fv.vol(pin, name='input patch')
            fv.vol(pout, name='output patch', opts=':colormap=lut:lut=nonlat')

            pred = model.predict(inb)

            pred2 = model2.predict(inb)
            aseg = np.argmax(pred2[0], axis=-1).squeeze()
            fv = fs.Freeview()
            fv.vol(inb[0].squeeze(), name='inb', opts=':locked=1')
            if use_subloss:
                padded_out = np.transpose(np.argmax(np.array(pred[1:1+len(patch_outputs)]), axis=-1).squeeze(), 
                                          (1,2,3,0))
                true_pad = np.transpose(np.argmax(np.array(ooutb[1:1+len(patch_outputs)]), axis=-1).squeeze(), 
                                        (1,2,3,0))

                fv.vol(true, name='padded unet target', opts=':colormap=lut:lut=nonlat:visible=0')
            else:
                padded_out = np.transpose(np.argmax(np.array(pred2[1:1+len(patch_outputs)]), axis=-1).squeeze(), 
                                          (1,2,3,0))
            fv.vol(np.argmax(outb[0].squeeze(), axis=-1), name='outb', opts=':colormap=lut:lut=nonlat:visible=0')
            fv.vol(aseg, name='aseg', opts=':colormap=lut:lut=nonlat:visible=0')
            fv.vol(padded_out, name='padded unet out', opts=':colormap=lut:lut=nonlat:visible=0')
            fv.show()
            assert 0
        keys = ['loss', 'val_loss'] if use_subloss else ['loss', 'val_loss']
        pfc([write_cb.fname], keys=keys, close_all=True, smooth=15, 
            remove_outlier_thresh=2, outlier_whalf=4, plot_block=False)
    

with tf.device(synth_device):
    vgen = gens.real_gen(mri_man_segs_recoded, mri_norms, vxm_model, norm_atlas, 
                         None, labels_in, subnet_patches=subnet_patches if use_subloss else None,
                         use_log_for_subnet=subloss == 'mse',
                         batch_size=1, use_rand=True, 
                         gpuid=synth_gpu, debug=False, add_outside=oshapes)
    gen = gens.synth_gen(mri_segs_recoded, gen_model, vxm_model, norm_atlas, 
                         None, labels_in, batch_size=1, use_rand=True, gpuid=synth_gpu, debug=False, 
                         use_log_for_subnet=subloss == 'mse',
                         subnet_patches=subnet_patches if use_subloss else None,
                         add_outside=oshapes)

ilist = []
olist = []
plist = []
dlist_sub = []
dlist_aseg = []
pulist = []
allist = []
patch_pred_list = []
choroid_label = target_lut.search('Left-Choroid')[0]
accumbens_label = target_lut.search('Left-Accumbens')[0]
mask = np.ones((nlabels_small,))
mask[lesion_label] = 0
mask[choroid_label] = 0
mask[accumbens_label] = 0
mask[0] = 0
lfunc_dice = nes.losses.DiceNonzero(nlabels_small, weights=None, check_input_limits=False).loss
lfunc_dice = ne.losses.Dice(nb_labels=nlabels_small, weights=None, check_input_limits=False).loss
ntest = 20
aseg_test_fscale = 1.75
aseg_test_fscale = 1.5
aseg_model_saved = tf.keras.models.load_model(f'aseg.fscale.{aseg_test_fscale}.h5', custom_objects=ld.layer_dict)
#aseg_fname = 'aseg.outside.unet_nf.64.21.0.levels.6.warp_max.2.oshapes.True.h5'
#aseg_model_saved.load_weights(aseg_fname)
dlist_sub_labels = []
dlist_aseg_labels = []
for n in tqdm(range(ntest)):
    with tf.device(model_device):
        inb, outb = next(vgen)
        pred = model.predict(inb)
        pred2 = aseg_model_saved.predict(inb)
        ppred = pmodel.predict(inb)
        if type(pred) is not list:
            pred = [pred]

        # d = model.evaluate(inb, outb, verbose=0)
        # d2 = aseg_model_saved.evaluate(inb, outb, verbose=0)
        d = lfunc_dice(tf.convert_to_tensor(outb[0], tf.float32), tf.convert_to_tensor(pred[0], tf.float32))
        d2 = lfunc_dice(tf.convert_to_tensor(outb[0], tf.float32), tf.convert_to_tensor(pred2, tf.float32))

    patch_pred_list.append(np.argmax(ppred[-1].squeeze(), axis=-1).copy())
    dlist_sub_labels.append(d.numpy())
    dlist_aseg_labels.append(d2.numpy())
    dlist_sub.append((d.numpy() * mask).sum() / mask.sum())
    dlist_aseg.append((d2.numpy() * mask).sum() / mask.sum())
    allist.append(np.argmax(pred2[0].squeeze(), axis=-1).copy)
    ilist.append(inb[0].squeeze().copy())
    olist.append(np.argmax(outb[0].squeeze(), axis=-1).copy())
    plist.append(np.argmax(pred[0].squeeze(), axis=-1).copy())


darray_sub = np.array(dlist_sub)
darray_aseg = np.array(dlist_aseg)
parray = np.concatenate([darray_aseg[np.newaxis], darray_sub[np.newaxis]])
xones = np.ones(parray.shape)
xones[0, :] *= 0
if 0:
    fig = plt.figure()
    plt.scatter(xones, parray)
    plt.xticks(ticks=[0, 1], labels=['synthseg', 'subseg'])
    plt.show(block=False)
else:
    fig, ax = plt.subplots()
    xp_aseg = np.ones(darray_aseg.shape)
    xp_sub = np.ones(darray_sub.shape)
    offset = .15
    lw = 2
    sub_props = {'color':'g', 'linewidth' : lw}
    aseg_props = {'color':'r', 'linewidth' : lw}
    aseg_bp = ax.boxplot(-darray_aseg, positions=[1-offset], sym='ro', 
                    boxprops=aseg_props, medianprops=aseg_props, whiskerprops=aseg_props, capprops=aseg_props)
    sub_bp = ax.boxplot(np.transpose(-darray_sub), positions=[1+offset], sym='gx', 
                    boxprops=sub_props, medianprops=sub_props, whiskerprops=sub_props, capprops=sub_props)
    ax.set_xticks([1-offset, 1+offset])
    xt = ax.xaxis.get_ticklabels()
    fontsize = 18
    fontweight = 'bold'
    fontproperties = {'weight' : fontweight, 'size' : fontsize}
    #ax.set_xticklabels(ax.get_xticks(), fontproperties)
    #ax.set_yticklabels(ax.get_yticks(), fontproperties)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight(fontweight)

    fdict = { 'fontsize' : fontsize, 'fontweight' : 'bold' }
    xt = ['SYNTHSEG', 'SUBSEG']
    ax.xaxis.set_ticklabels(xt)
    ax.tick_params(labelsize=fontsize)
    plt.xlabel("segmentation method", fontdict=fdict)
    plt.ylabel("Dice Coefficient", fontdict=fdict)
    plt.title(f'comparison of segmentation methods (N={len(darray_sub)})', fontdict=fdict)
    yl = ax.get_ylim()
    if 0:
        leg = plt.legend(['ASEG', 'SUBSEG'])
        leg.legendHandles[0].set_color('red')
        leg.legendHandles[1].set_color('green')

    print(f'SUBNET: real dice {np.array(dlist_sub).mean()}')
    #print(f'{dlist_sub}')
    
    print(f'SYNTHSEG: real dice {np.array(dlist_aseg).mean()}')
    #print(f'{dlist_aseg}')
    print(f'SUBNET-ASEG: real dice {-np.array(dlist_sub).mean()+np.array(dlist_aseg).mean()}')

    plt.show(block=True)



imgs = np.array(ilist)
tseg = np.array(olist)
pseg = np.array(plist)
patch_out = np.array(patch_pred_list)
fv = fs.Freeview(swap_batch_dim=True)
fv.vol(imgs, name='img', opts=':locked=1:linked=1')
fv.vol(patch_out, name='patch seg', opts=':colormap=lut:visible=0:linked=1:locked=1', lut=target_lut)
fv.vol(tseg, name='true seg', opts=':colormap=lut:visible=0:linked=1', lut=target_lut)
fv.vol(pseg, name='pred seg', opts=':colormap=lut:visible=1:linked=1', lut=target_lut)
fv.show()
#pfc(write_cb.fname, keys=['loss'], close_all=True, smooth=15, remove_outlier_thresh=2,
#        outlier_whalf=4, plot_block=False)




if 0:

    a1 = .1*np.ones(pred.shape[0:-1] + (5,))
    a1[..., 0] = pred[..., 0]  # bg
    a1[..., 1] = pred[..., 2]  # gm
    a1[..., 2] = pred[..., 6] # thalamus
    a1[..., 3] = pred[..., 11] # hippo
    lab_to_ind = -1 * np.ones((pred.shape[-1],), int)
    lab_to_ind[0] = 0
    lab_to_ind[2] = 1
    lab_to_ind[6] = 2
    lab_to_ind[11] = 3

    ind_to_lab = np.zeros((a1.shape[-1],), int)
    for ind in range(len(lab_to_ind)):
        if lab_to_ind[ind] >= 0:
            ind_to_lab[int(lab_to_ind[ind])] = ind

    ind_mat = np.zeros((a1.shape[-1],pred.shape[-1]))
    for ind, lab in enumerate(ind_to_lab):
        ind_mat[ind, lab] = 1

    p2 = tf.matmul(a1, ind_mat)
    fs.fv(inb, np.argmax(pred, axis=-1), np.argmax(a1, axis=-1), np.argmax(p2, axis=-1))


if 0:
    import surfa
    vmp = surfa.system.vmpeak()

    fn1 = 'subnets.outside.unet_nf.64.warp_max.2.oshapes.True.num_subnets.15.psize.24.lab2ind.True.lr.0.0001.subloss.True.txt'
    fn2 = 'subnets.outside.unet_nf.64.warp_max.2.oshapes.True.num_subnets.15.psize.24.lab2ind.True.lr.1e-05.subloss.True.txt'
    fn3 = 'subnets.outside.unet_nf.64.warp_max.2.oshapes.True.num_subnets.15.psize.24.lab2ind.False.lr.0.0001.subloss.True.txt'
    fn4 = 'subnets.outside.unet_nf.64.warp_max.2.oshapes.True.num_subnets.15.psize.24.lab2ind.True.lr.0.0001.subloss.False.txt'
    pfc(
        [fn1, fn2, fn3, fn4],
        legend=['lab2ind lr4', 'lab2ind lr5', 'no lab2ind', 'no subloss'],
        keys=["seg_loss", "val_seg_loss"],
        close_all=True,
        smooth=15,
        remove_outlier_thresh=2,
        outlier_whalf=4,
        plot_block=False)


    pfc(
        [fn1, fn2, fn3],
        legend=['lab2ind lr4', 'lab2ind lr5', 'no lab2ind'],
        keys=["subloss_loss", "val_subloss_loss"],
        close_all=True,
        smooth=15,
        remove_outlier_thresh=2,
        outlier_whalf=4,
        plot_block=False)


    fn1 = 'subnets.outside.unet_nf.64.warp_max.2.oshapes.True.num_subnets.60.psize.24.pad.8.lab2ind.True.lr.0.0001.subloss.True.combined_training.True.txt'
    fn2 = 'subnets.outside.unet_nf.64.warp_max.2.oshapes.True.num_subnets.90.psize.16.pad.8.lab2ind.True.lr.0.0001.subloss.True.combined_training.True.txt'
    fn3 = 'subnets.outside.unet_nf.64.warp_max.2.oshapes.True.num_subnets.128.psize.12.pad.8.lab2ind.True.lr.0.0001.subloss.True.combined_training.True.txt'
    pfc([fn1, fn2, fn3], plot_block=True, smooth=5, keys=['loss', 'seg_loss', 'subloss_loss', 'val_seg_loss'],
        legend=['n60p24','n90p16', 'n128p12'])


if 0:
    po = nes.layers.ExtractPatch(((p0, p1), (p0, p1), (p0, p1)),
                                 name=f'subnet_inp{subnet_no}')(patch_input)
    pb = nes.layers.Pad(padding=padding, mode='constant', name=f'pad_io{subnet_no}')(po)
    m = tf.keras.Model(model.inputs, [pb, aseg_model.inputs[0], subnet_out])
    p = m.predict(inb)
    p[1][0, x0, y0, z0, :]


    fs.fv(p[1], p[0])


    # m = tf.keras.Model(model.inputs, [subnet_out, unet_out])
    m = tf.keras.Model(model.inputs, [patch_input,  summed_patch_outputs, aseg_linear_out])
    p = m.predict(inb)
    p1 = model.predict(inb)
    # fs.fv(np.pad(p[0].squeeze(), ((pad_size, pad_size),)*3, mode='constant'), np.argmax(p[1], axis=-1).squeeze())
    x0, y0, z0 = 88, 151, 35
    p[1][0, x0, y0, z0, :]
    p[2][0, x0, y0, z0, :]

    px0 = 14
    py0 = 7
    pz0 = 20

if 0:
    f_I_nL_4 = 'subnets.outside.unet_nf.60.warp_max.2.oshapes.True.num_subnets.16.psize.32.pad.8.lab2ind.False.lr.0.0001.subloss.False.insertion.True..combined_training.False.train_aseg.False.txt'
    f_I_nL_5 = 'subnets.outside.unet_nf.60.warp_max.2.oshapes.True.num_subnets.16.psize.32.pad.8.lab2ind.False.lr.1e-05.subloss.False.insertion.True..combined_training.False.train_aseg.False.txt'

    f_I_L_5 = 'subnets.outside.unet_nf.60.warp_max.2.oshapes.True.num_subnets.16.psize.32.pad.8.lab2ind.True.lr.1e-05.subloss.False.insertion.True..combined_training.False.train_aseg.False.txt'

    f_nI_L_5 = 'subnets.outside.unet_nf.60.warp_max.2.oshapes.True.num_subnets.16.psize.32.pad.8.lab2ind.True.lr.1e-05.subloss.False.insertion.False..combined_training.False.train_aseg.False.txt'

    pfc([f_I_nL_4, f_I_nL_5, f_I_L_5, f_nI_L_5], keys=['loss', 'val_loss'], close_all=True, smooth=15, remove_outlier_thresh=2, outlier_whalf=4, plot_block=True, legend=['ins_nolab2ind_4', 'ins_nolab2ind_5', 'no ins_lab2ind_5', 'no ins_lab2ind_5'])

    f_nI_nL_5 = 'subnets.outside.unet_nf.60.warp_max.2.oshapes.True.num_subnets.1.psize.32.pad.8.lab2ind.False.lr.1e-05.subloss.False.insertion.True..combined_training.False.train_aseg.False.txt'
    f_nI_L_5 = 'subnets.outside.unet_nf.60.warp_max.2.oshapes.True.num_subnets.1.psize.32.pad.8.lab2ind.True.lr.1e-05.subloss.False.insertion.False..combined_training.False.train_aseg.False.txt'
    f_nI_nL_4 = 'subnets.outside.unet_nf.60.warp_max.2.oshapes.True.num_subnets.1.psize.32.pad.8.lab2ind.False.lr.0.0001.subloss.False.insertion.False..combined_training.False.train_aseg.False.txt'
    f_I_L_4 = 'subnets.outside.unet_nf.60.warp_max.2.oshapes.True.num_subnets.1.psize.32.pad.8.lab2ind.False.lr.0.0001.subloss.False.insertion.True..combined_training.False.train_aseg.False.txt'

    pfc([f_nI_nL_5, f_nI_L_5, f_nI_nL_4, f_I_L_4], keys=['loss', 'val_loss'], close_all=True, smooth=15, remove_outlier_thresh=2, outlier_whalf=4, plot_block=True, legend=['no ins_nolab_5', 'no ins_lab_5', 'no ins_no lab_4', 'ins_lab_4'])

    [fc, fnc, fna, fnac, f48] = ['subnets.outside.unet_nf.60.warp_max.2.oshapes.True.num_subnets.16.psize.32.pad.8.lab2ind.False.lr.0.0001.subloss.True.insertion.True.combined_training.False.train_aseg.False.txt',
 'subnets.outside.unet_nf.60.warp_max.2.oshapes.True.num_subnets.16.psize.32.pad.8.lab2ind.False.lr.0.0001.subloss.True.insertion.True.combined_training.False.train_aseg.False.concat_aseg.False.txt',
 'subnets.outside.unet_nf.60.warp_max.2.oshapes.True.num_subnets.16.psize.32.pad.8.lab2ind.False.lr.0.0001.subloss.True.insertion.True.combined_training.False.train_aseg.True.concat_aseg.False.txt',
 'subnets.outside.unet_nf.60.warp_max.2.oshapes.True.num_subnets.16.psize.32.pad.8.lab2ind.False.lr.0.0001.subloss.True.insertion.True.combined_training.False.train_aseg.True.txt',
 'subnets.outside.unet_nf.60.warp_max.2.oshapes.True.num_subnets.48.psize.32.pad.8.lab2ind.False.lr.0.0001.subloss.True.insertion.True.combined_training.False.train_aseg.True.txt']
    pfc([fc, fnc, fna, fnac, f48], keys=['loss', 'seg_loss', 'subloss_loss', 'val_subloss_loss', 'val_seg_loss'], close_all=True, smooth=51, remove_outlier_thresh=2,        outlier_whalf=4, plot_block=False, legend=['concat', 'no concat', 'train aseg', 'train_concat', 'N48'])
