import numpy as np
import os
from tensorflow.keras import layers as KL
import tensorflow as tf
from tensorflow.keras import backend as K

import pdb as gdb
import neurite as ne
import neurite_sandbox as nes
import voxelmorph as vxm
import voxelmorph_sandbox as vxms

def read_vxm_model(inshape):

    f = 128
    conf = {
        'def.enc_nf': [f] * 4,
        'def.dec_nf': [f] * 4,
        'def.add_nf': [f] * 4,
    'def.hyp_den': [32] * 4,
    }
    
    vxm_model = vxms.networks.VxmJointAverage(in_shape=inshape, **conf)
    vxm_model.load_weights(os.path.join('models_from_Malte', f'VxmJointAverage{f}.h5'))

    return vxm_model

def minmax_norm(x):
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min)
    
def create_gen_model(np_segs, oshapes, synth_device, nlabels_small, labels_in, inshape, warp_max=2,
                     zero_max=1, gen_args=None, doaff=False,t1=False):
    # these control how much the labels are warped to create a new "subject"
    warp_min=.5
    warp_blur_min=np.array([2, 4, 8])
    warp_blur_max=warp_blur_min*2
    bias_blur_min=np.array([2, 4, 8])
    bias_blur_max=bias_blur_min*2
    if t1:
        warp_min = warp_max = 0

    print(f'using warp max = {warp_max} and nlabels {nlabels_small}')
    if gen_args is None:
        gen_args = dict(
            aff_shift=doaff * 20,
            aff_rotate=doaff * 30,
            aff_scale=doaff * .1,
            warp_min=warp_min,
            warp_max=warp_max,
            blur_max=2,  # was .5, then 1
            bias_max=.25,  # was 2 then .5
            bias_blur_min=bias_blur_min,
            bias_blur_max=bias_blur_max,
            gamma=0,
            # warp_zero_mean=True,
            zero_background=.95,
            noise_max=.2,
            noise_min=.1  #    clip_max=2800,
        )

    with tf.device(synth_device):
        if oshapes:
            num_outside_shapes_to_add = 8
            l2l = nes.models.labels_to_labels(
                labels_in,
                shapes_num=num_outside_shapes_to_add,
                shapes_zero_max=zero_max,
                in_shape=inshape,
                shapes_add=True
            )
            nlabels = labels_in.max()+1
            labels_in_with_shapes = np.concatenate([labels_in, np.arange(nlabels, 
                                                                         nlabels+2+num_outside_shapes_to_add)])
            labels_out = labels_in_with_shapes.copy()
            labels_out[labels_out >= nlabels] = 0  # don't include outside shapes in one-hots
            lout_dict = dict()
            for i in range(len(labels_out)):
                lout_dict[i] = labels_out[i]

            gen_model0 = nes.models.labels_to_image(labels_in=labels_in_with_shapes, labels_out=lout_dict, 
                                                    in_shape=inshape, **gen_args, num_chan=1, id=1)
            new_out = gen_model0(l2l.outputs)
            gen_model = tf.keras.models.Model(l2l.inputs, new_out)
        else:
            gen_model = nes.models.labels_to_image(labels_in=labels_in, in_shape=inshape, **gen_args, num_chan=1, id=1)

    return gen_model


def synth_gen(label_vols, gen_model_orig, vxm_model, norm_atlas, lab_to_ind, labels_in, batch_size=1, 
              use_rand=True, gpuid=1, seg_resize=1, num_outside_shapes_to_add=8, use_log=False, debug=False, t1=False,
              add_outside=True, smooth_wt=0.5,
              zero_background=.1,
              use_log_for_subnet=False,
              subnet_patches=None
          ):

    if t1:
        
        while True:
            adir = '/autofs/cluster/vxmdata1/FS_Slim/proc/cleaned'
            mname = 'aseg.mgz'
            vname = 'norm.mgz'
            man_subjects = [
                f for f in os.listdir(adir)
                if os.path.isdir(os.path.join(adir, f)) and os.path.isfile(os.path.join(adir, f, mname))
            ]
            mri_man_segs = []  # manual segs
            mri_norms = []  # mri vols
            
            man_subjects = random.sample(man_subjects, 1)
        
            mri_seg_orig = sf.load_volume(os.path.join(adir, s, mname))
            mri_man_segs_orig.append(mri_seg_orig)
            mri_seg = mri_seg_orig.reshape(target_shape)
            mri_man_segs.append(mri_seg)
            
            mri_norm_orig = sf.load_volume(os.path.join(adir, s,vname))
            mri_norm = mri_norm_orig.resample_like(mri_seg)
            mri_norms.append(mri_norm)
            # mri_norms_orig.append(mri_norm_orig)
            yield mri_norms, mri_man_segs
        
    gen_model = gen_model_orig
    nlabels = gen_model.outputs[-1].get_shape().as_list()[-1]  # number of compressed labels
    if vxm_model is not None:
        batch_smooth_wt = np.zeros((batch_size, 1))
        batch_smooth_wt[:,0] = smooth_wt
        atlas_input = KL.Input(norm_atlas.shape[1:])
        smooth_wt_input = KL.Input(vxm_model.inputs[0].shape[1:])
        batch_norm_atlas = np.zeros((batch_size,) + norm_atlas.shape[1:])
        batch_norm_atlas[:,...] = norm_atlas
        mask = KL.Lambda(lambda x: tf.cast(tf.greater(tf.argmax(x, axis=-1), 0), x.dtype))(
            gen_model.outputs[1])
        warp_input = KL.Multiply(name='mask')([mask[..., tf.newaxis], gen_model.outputs[0]])
        transform = vxm_model.outputs[0] # vxm_model(vxm_model.inputs)
        inputs = [.5*tf.ones(vxm_model.inputs[0].shape[1:]), gen_model.outputs[0], atlas_input]
        inputs = [smooth_wt_input, warp_input, atlas_input]
        transform = vxm_model(inputs)
        warped_aseg = vxm.layers.SpatialTransformer(interp_method='linear', fill_value=0, name='warp_aseg')(
            [gen_model.outputs[1], transform])
        warped_im = vxm.layers.SpatialTransformer(interp_method='linear', fill_value=0, name='warp_im')(
            [gen_model.outputs[0], transform])
        gen_model = tf.keras.models.Model(gen_model.inputs + [smooth_wt_input, atlas_input], 
                                          [warped_im, warped_aseg])

    # transform = vxm_model.predict([l, norm, norm_atlas])
    inshape = label_vols[0].shape

    if subnet_patches is not None:
        num_subnets = len(subnet_patches)
        sub_shape = ((subnet_patches[0][0][1] - subnet_patches[0][0][0]),)*3
        batch_subnets = np.zeros((batch_size, num_subnets, *sub_shape, nlabels))

    if 0 and add_outside:
        l2l = nes.models.labels_to_labels(
            labels_in,
            shapes_num=num_outside_shapes_to_add,
            in_shape=label_vols[0].shape,
            shapes_add=True
        )
        li = np.concatenate([labels_in, np.arange(labels_in.max()+1, 
                                                  labels_in.max()+1+num_outside_shapes_to_add)])
        l2i = nes.models.labels_to_image(
            labels_in=li,
            labels_out=None,
            in_shape=label_vols[0].shape,
            zero_background=.5,  #  was 1
            noise_max=.2,
            noise_min=.1,
            warp_max=0
        )

    # outputs [6] and [7] are the t2 labels without (6) and with (7) atrophy
    batch_input_labels = np.zeros((batch_size, *inshape, 1))
    label_shape = tuple(np.array(inshape) // seg_resize)
    batch_onehots = np.zeros((batch_size, *label_shape, nlabels))
    batch_images = np.zeros((batch_size, *inshape, 1))

    if debug:
        batch_orig_images1 = np.zeros((batch_size, *inshape, 1))
        batch_orig_images2 = np.zeros((batch_size, *inshape, 1))
        batch_orig_labels1 = np.zeros((batch_size, *label_shape, 1))
        batch_orig_labels2 = np.zeros((batch_size, *label_shape, 1))

    if gpuid >= 0:
        device = '/gpu:' + str(gpuid)
    else:
        device = '/physical_device:CPU:0'
        device = '/cpu:0'

    ind = -1
    bg_onehot = np.zeros((nlabels,))
    bg_onehot[0] = 1
    while (True):
        for bind in range(batch_size):
            if use_rand:
                ind = np.random.randint(0, len(label_vols))
            else:
                ind = np.mod(ind+1, len(label_vols))
            # print(label_vols[ind].data.shape,batch_input_labels[bind,...].shape)
            # batch_input_labels[bind,...] = label_vols[ind].data[...,np.newaxis]
            batch_input_labels[bind,...] = np.expand_dims(label_vols[ind].data, axis=-1)

        with tf.device(device):
            print("batch inpu:",batch_input_labels.shape, " batch norm atlas:", batch_norm_atlas.shape)
            if len(gen_model.inputs) > 1:
                pred = gen_model.predict_on_batch([batch_input_labels, batch_smooth_wt, batch_norm_atlas])
            else:
                pred = gen_model.predict_on_batch(batch_input_labels)

        for bind in range(batch_size):
            im = pred[0][bind,...]
            onehot = pred[1][bind,...]
            onehot[np.argmax(onehot,axis=-1) == 0] = bg_onehot   # not sure why I have to do this by bg is all 0

            seg = tf.argmax(onehot, axis=-1)
            if 0 and np.random.rand() < zero_background:
                mask = tf.cast(mask > 0, im.dtype)
                im *= mask

            # if add_outside:
                # im = nes.utils.augment.add_outside_shapes(im[..., 0], np.argmax(onehot, axis=-1), labels_in, l2l=l2l, l2i=l2i)[..., np.newaxis]

            if use_log:
                onehot[onehot == 0] = -10
                onehot[onehot == 1] = 10

            batch_images[bind, ...] = im
            batch_onehots[bind, ...] = onehot
            if subnet_patches is not None:  # return subnet training targets
                for sno in range(len(subnet_patches)):
                    x0 = subnet_patches[sno][0][0]
                    x1 = subnet_patches[sno][0][1]
                    y0 = subnet_patches[sno][1][0]
                    y1 = subnet_patches[sno][1][1]
                    z0 = subnet_patches[sno][2][0]
                    z1 = subnet_patches[sno][2][1]
                    batch_subnets[bind, sno, ...] = onehot[x0:x1, y0:y1, z0:z1, ...]
                    if use_log_for_subnet:
                        batch_subnets[batch_subnets > 0] = 10
                        batch_subnets[batch_subnets == 0] = -10

                        

        inputs = [batch_images]
        outputs = [batch_onehots]
        if subnet_patches is not None:  # return subnet training targets
            outputs += [batch_subnets]

        yield inputs, outputs
                
    return 0


def minmax_norm(x):
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min)

def real_gen(label_vols, norm_vols, vxm_model, norm_atlas, lab_to_ind, labels_in, batch_size=1, 
             use_rand=True, gpuid=3, 
             use_log_for_subnet=False,
             seg_resize=1, num_outside_shapes_to_add=8, use_log=False, debug=False, 
             subnet_patches=None,
             add_outside=False, smooth_wt=0.5):

    if gpuid >= 0:
        device = '/gpu:' + str(gpuid)
    else:
        device = '/physical_device:CPU:0'
        device = '/cpu:0'

    with tf.device(device):
        nlabels = labels_in.max() + 1  # number of compressed labels
        aseg_input = KL.Input(label_vols[0].shape + (nlabels,), name='aseg_in')
        transform = vxm_model.outputs[0] # vxm_model(vxm_model.inputs)

    inshape = label_vols[0].shape
    batch_smooth_wt = np.zeros((1, 1))
    batch_smooth_wt[:,0] = .5

    # outputs [6] and [7] are the t2 labels without (6) and with (7) atrophy
    batch_input_labels = np.zeros((batch_size, *inshape, 1))
    label_shape = tuple(np.array(inshape) // seg_resize)
    batch_onehots = np.zeros((batch_size, *label_shape, nlabels))
    batch_images = np.zeros((batch_size, *inshape, 1))

    if subnet_patches is not None:
        num_subnets = len(subnet_patches)
        sub_shape = ((subnet_patches[0][0][1] - subnet_patches[0][0][0]),)*3
        batch_subnets = np.zeros((batch_size, num_subnets, *sub_shape, nlabels))

    ind = -1
    while (True):
        for bind in range(batch_size):
            if use_rand:
                ind = np.random.randint(0, len(label_vols))
            else:
                ind = np.mod(ind+1, len(label_vols))
            seg = np.expand_dims(label_vols[ind].data,axis=-1)#[..., np.newaxis]
            batch_input_labels[bind,...] = seg

            im = np.expand_dims(norm_vols[ind].data,axis=-1)#[..., np.newaxis]
            # im /= im.max()
            im = minmax_norm(im)

            onehot = np.eye(nlabels)[label_vols[ind].data]

            if use_log:
                onehot[onehot == 0] = -10
                onehot[onehot == 1] = 10

            with tf.device(device):

                seg_mask = tf.cast(seg > 0, im.dtype)#[..., tf.newaxis]
                warp_input = im * seg_mask
                print(im.shape,seg_mask.shape)
                inputs = [batch_smooth_wt, im[np.newaxis,...], norm_atlas, onehot[np.newaxis, ...]]

                im_warped = im[np.newaxis,...]
                onehot_warped = onehot[np.newaxis, ...]
                onehot_warped[onehot_warped > 1] = 1


            batch_images[bind, ...] = im_warped[0]
            batch_onehots[bind, ...] = onehot_warped[0]
            if subnet_patches is not None:  # return subnet training targets
                for sno in range(len(subnet_patches)):
                    x0 = subnet_patches[sno][0][0]
                    x1 = subnet_patches[sno][0][1]
                    y0 = subnet_patches[sno][1][0]
                    y1 = subnet_patches[sno][1][1]
                    z0 = subnet_patches[sno][2][0]
                    z1 = subnet_patches[sno][2][1]
                    batch_subnets[bind, sno, ...] = onehot_warped[0][x0:x1, y0:y1, z0:z1, ...]


        inputs = [batch_images]
        outputs = [batch_onehots]
        if subnet_patches is not None:  # return subnet training targets
            if use_log_for_subnet:
                batch_subnets[batch_subnets > 0] = 10
                batch_subnets[batch_subnets == 0] = -10

            outputs += [batch_subnets]

        yield inputs, outputs
                
    return 0
def real_val(label_vols, norm_vols, vxm_model, norm_atlas, lab_to_ind, labels_in, batch_size=1, 
             use_rand=True, gpuid=3, 
             use_log_for_subnet=False,
             seg_resize=1, num_outside_shapes_to_add=8, use_log=False, debug=False, 
             subnet_patches=None,
             add_outside=False, smooth_wt=0.5):

    if gpuid >= 0:
        device = '/gpu:' + str(gpuid)
    else:
        device = '/physical_device:CPU:0'
        device = '/cpu:0'

    with tf.device(device):
        nlabels = labels_in.max() + 1  # number of compressed labels
        aseg_input = KL.Input(label_vols[0].shape + (nlabels,), name='aseg_in')
        transform = vxm_model.outputs[0] # vxm_model(vxm_model.inputs)
        warped_aseg = vxm.layers.SpatialTransformer(interp_method='linear', fill_value=0)([aseg_input, transform])
        warped_im = vxm.layers.SpatialTransformer(interp_method='linear', fill_value=0)(
            [vxm_model.inputs[1], transform])
        warp_model = tf.keras.models.Model(vxm_model.inputs + [aseg_input], [warped_im, warped_aseg])

    # transform = vxm_model.predict([l, norm, norm_atlas])
    inshape = label_vols[0].shape
    batch_smooth_wt = np.zeros((1, 1))
    batch_smooth_wt[:,0] = .5

    # outputs [6] and [7] are the t2 labels without (6) and with (7) atrophy
    batch_input_labels = np.zeros((batch_size, *inshape, 1))
    label_shape = tuple(np.array(inshape) // seg_resize)
    batch_onehots = np.zeros((batch_size, *label_shape, nlabels))
    batch_images = np.zeros((batch_size, *inshape, 1))

    if subnet_patches is not None:
        num_subnets = len(subnet_patches)
        sub_shape = ((subnet_patches[0][0][1] - subnet_patches[0][0][0]),)*3
        batch_subnets = np.zeros((batch_size, num_subnets, *sub_shape, nlabels))

    ind = -1
    while (True):
        for bind in range(batch_size):
            if use_rand:
                ind = np.random.randint(0, len(label_vols))
            else:
                ind = np.mod(ind+1, len(label_vols))
            seg = np.expand_dims(label_vols[ind].data,axis=-1)#[..., np.newaxis]
            batch_input_labels[bind,...] = seg

            im = np.expand_dims(norm_vols[ind].data,axis=-1)#[..., np.newaxis]
            # im /= im.max()
            im = minmax_norm(im)

            onehot = np.eye(nlabels)[label_vols[ind].data]

            if use_log:
                onehot[onehot == 0] = -10
                onehot[onehot == 1] = 10

            with tf.device(device):

                # Apply the mask and perform the multiplication
                seg_mask = tf.cast(seg > 0, im.dtype)[..., tf.newaxis]
                warp_input = im * seg_mask

                inputs = [batch_smooth_wt, im[np.newaxis,...], norm_atlas, onehot[np.newaxis, ...]]
                im_warped, onehot_warped = warp_model.predict(inputs)
                onehot_warped[onehot_warped > 1] = 1

            batch_images[bind, ...] = im_warped[0]
            batch_onehots[bind, ...] = onehot_warped[0]
            if subnet_patches is not None:  # return subnet training targets
                for sno in range(len(subnet_patches)):
                    x0 = subnet_patches[sno][0][0]
                    x1 = subnet_patches[sno][0][1]
                    y0 = subnet_patches[sno][1][0]
                    y1 = subnet_patches[sno][1][1]
                    z0 = subnet_patches[sno][2][0]
                    z1 = subnet_patches[sno][2][1]
                    batch_subnets[bind, sno, ...] = onehot_warped[0][x0:x1, y0:y1, z0:z1, ...]


        inputs = [batch_images]
        outputs = [batch_onehots]
        if subnet_patches is not None:  # return subnet training targets
            if use_log_for_subnet:
                batch_subnets[batch_subnets > 0] = 10
                batch_subnets[batch_subnets == 0] = -10

            outputs += [batch_subnets]

        yield inputs, outputs
                
    return 0
