import neurite as ne
import neurite_sandbox as nes
import voxelmorph as vxm
import tensorflow as tf

    # 'ExtractPatch' : nes.layers.ExtractPatch,
    # 'InsertPatch' : nes.layers.InsertPatch,

layer_dict = {
    'LocalParamWithInput' : ne.layers.LocalParamWithInput,
    'RescaleTransform' : vxm.layers.RescaleTransform,
    'VecInt' : vxm.layers.VecInt,
    'SpatialTransformer' : vxm.layers.SpatialTransformer,
    'ComposeTransform' : vxm.layers.ComposeTransform,
    'MeanStream' : ne.layers.MeanStream,
    'mean_loss' : ne.losses.Dice().mean_loss,
    'loss' : vxm.losses.Grad(penalty='l2').loss,
    'Negate' : ne.layers.Negate,
    'Pad' : nes.layers.Pad,
    'AffineToDenseShift' : vxm.layers.AffineToDenseShift,
    'DrawAffineParams' : vxm.layers.DrawAffineParams,
    'ParamsToAffineMatrix' : vxm.layers.ParamsToAffineMatrix,
    'mean_loss' : ne.losses.Dice.mean_loss,
    'loss' : tf.keras.losses.mse
}
