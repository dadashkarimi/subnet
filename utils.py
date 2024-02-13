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


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, base_log_dir, **kwargs):
         super(CustomTensorBoard, self).__init__(**kwargs)
         self.base_log_dir = base_log_dir

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % 100 == 0:  # Check if it's the start of a new set of 50 epochs
            self.log_dir = f"{self.base_log_dir}/epoch_{epoch}"
            super().set_model(self.model)
