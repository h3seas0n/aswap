from keras.models import load_model
import numpy as np
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

import tensorflow as tf

"""
Function for creating heatmap (CAM method)
"""

def make_gradcam_heatmap(img_array, model):
    # model without labelling layers
    last_conv_layer = model.get_layer("block14_sepconv2_act")
    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)

    # only the labelling layers
    classifier_input = Input(shape=last_conv_layer.output.shape[1:])
    x = model.get_layer("avg_pool")(classifier_input)
    x = model.get_layer("predictions")(x)
    classifier_model = Model(classifier_input, x)

    # we calculate the gradient for the highest probability class with respect to the last conv layer
    with tf.GradientTape() as tape:  # automatic differentiation
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)

        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(
        top_class_channel, last_conv_layer_output)  # compute gradient

    # the mean intensity of the gradient over a feature ("how important each channel is")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # we multiply each channel by it's importance
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # create heatmap
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # heatmap is normalized between 0 and 1
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap))
    return heatmap
