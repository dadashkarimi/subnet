import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
import neurite as ne
from neurite_sandbox.tf.models import labels_to_labels
from neurite_sandbox.tf.utils.augment import add_outside_shapes
from neurite.tf.utils.augment import draw_perlin_full
import voxelmorph as vxm
import os
import glob

class PeriodicWeightsSaver(tf.keras.callbacks.Callback):
    def __init__(self, filepath, save_freq=200, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        # Save the weights every `save_freq` epochs
        if (epoch + 1) % self.save_freq == 0:
            weights_path = os.path.join(self.filepath, f"weights_epoch_{epoch + 1}.h5")
            self.model.save_weights(weights_path)
            print(f"Saved weights to {weights_path}")

def make_cmap(num_to_col=None, name='freesurfer'):
    '''Create Matplotlib colormap and normalization from FreeSurfer LUT.'''
    if not isinstance(num_to_col, dict):
        _, _, num_to_col = read_lut(num_to_col)
    import matplotlib.colors as mc
    num_col = max(num_to_col, key=int)
    colors = np.zeros((num_col + 1, 3))
    for k, v in num_to_col.items():
        colors[k, ...] = v / 255
    cmap = mc.ListedColormap(colors, name=name)
    norm = mc.Normalize(vmin=0, vmax=cmap.N-1)
    return cmap, norm

def read_lut(lut=None):
    '''Read FS lookup table (LUT) from LUT, FS directory or FREESURFER_HOME.'''
    fs = 'FREESURFER_HOME'
    assert fs in os.environ or os.path.exists(lut)
    if not lut:
        lut = os.environ[fs]
    if os.path.isdir(lut):
        lut = os.path.join(lut, 'FreeSurferColorLUT.txt')
    with open(lut, 'r') as f:
        lines = f.read().splitlines()
    lines = [l for l in (l.strip() for l in lines) if l and l[0] != '#']
    words = [w for w in (l.split() for l in lines)]
    num_to_name = {int(w[0]): w[1] for w in words}
    name_to_num = {v: k for k, v in num_to_name.items()}
    num_to_col = {int(w[0]): np.uint8(w[2:5]) for w in words}
    return num_to_name, name_to_num, num_to_col

def minmax_norm(mri, axis=None):
    """
    Min-max normalize an mri struct using a safe division.

    Arguments:
        x: np.array to be normalized
        axis: Dimensions to reduce during normalization. If None, all axes will be considered,
            treating the input as a single image. To normalize batches or features independently,
            exclude the respective dimensions.

    Returns:
        Normalized tensor.
    """
    x = mri.data
    x_min = x.min()
    x_max = x.max()
    mri.data = (x - x_min) / (x_max - x_min) 
    return mri



def map_labels(label):
    if label.startswith('Left-'):
        return label.replace('Left-', '')
    elif label.startswith('Right-'):
        return label.replace('Right-', '')
    else:
        return label

def unify_left_right(atlas):
    label_map = {}
    with open('synthseg-labels.txt', 'r') as file:
        next(file)  # Skip the header
        for line in file:
            line = line.strip().split()
            label_map[int(line[0])] = map_labels(line[1])

    reverse_label_map = {v: k for k, v in label_map.items()}
    
    # Initialize a dictionary to store the smallest value for each label
    smallest_label_values = {}
    
    # Update the smallest value dictionary
    for label, name in label_map.items():
        if name not in smallest_label_values or label < smallest_label_values[name]:
            smallest_label_values[name] = label
    
    for label, new_label in smallest_label_values.items():
        atlas[atlas == reverse_label_map[label]] = new_label
    return atlas
    
def my_nonzero_hard_dice(y_true, y_pred):
    unique_labels = np.unique(y_true)
    dice_scores = []
    for label in unique_labels:
        if label != 0:  # Exclude background label
            y_true_label = (y_true == label).astype(int)
            y_pred_label = (y_pred == label).astype(int)
            dice = dice_coefficient(y_true_label, y_pred_label)
            dice_scores.append(dice)
    if len(dice_scores) > 0:
        return np.mean(dice_scores)
    else:
        return 0.0  # If there are no non-zero labels, return 0
        
def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2.0 * intersection) / (union) 

def my_hard_dice(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    dice = dice_coefficient(y_true_flat, y_pred_flat)
    return dice
    
class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, base_log_dir, **kwargs):
         super(CustomTensorBoard, self).__init__(**kwargs)
         self.base_log_dir = base_log_dir

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % 100 == 0:  # Check if it's the start of a new set of 50 epochs
            self.log_dir = f"{self.base_log_dir}/epoch_{epoch}"
            super().set_model(self.model)

class val_loss:
    def __init__(self, lfunc, label_mask):
        self.lfunc = lfunc
        self.label_mask = label_mask

    def loss(yt, yp):
        lvals = self.lfunc(yt, yp)
        lval = tf.reduce_sum(lvals * self.label_mask) / tf.reduce_sum(self.label_mask)
        return lval
        
def synth_gen(label_vols, gen_model, lab_to_ind, labels_in, batch_size=8, use_rand=True, gpuid=1, 
              seg_resize=1, num_outside_shapes_to_add=8, use_log=False, debug=False, 
              add_outside=True):

    inshape = label_vols[0].shape
    nlabels = gen_model.outputs[-1].get_shape().as_list()[-1]  # number of compressed labels

    if add_outside:
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
    while (True):
        for bind in range(batch_size):
            if use_rand:
                ind = np.random.randint(0, len(label_vols))
            else:
                ind = np.mod(ind+1, len(label_vols))
            batch_input_labels[bind,...] = label_vols[ind].data[...,np.newaxis]

        with tf.device(device):
            pred = gen_model.predict_on_batch(batch_input_labels)

        for bind in range(batch_size):
            im = pred[0][bind,...]
            onehot = pred[1][bind,...]
            # if add_outside:
                # im = nes.utils.augment.add_outside_shapes(im[..., 0], np.argmax(onehot, axis=-1), labels_in, l2l=l2l, l2i=l2i)[..., np.newaxis]

            if use_log:
                onehot[onehot == 0] = -10
                onehot[onehot == 1] = 10

            batch_images[bind, ...] = im
            batch_onehots[bind, ...] = onehot

        inputs = [batch_images]
        outputs = [batch_onehots]

        yield inputs, outputs
                
    return 0

