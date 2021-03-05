import tensorflow as tf
import matplotlib.cm as cm
import PIL
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img, random_rotation, random_zoom, random_shift
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception, preprocess_input
import numpy as np
from keras.models import load_model
import utilsDetection
import streamlit as st

"""
Detection model integration on our JS-based website (aquation.ml) - live detection function disabled due to high cost
"""

st.markdown(
    "<style>.st-cf{color:#6EC1E4!important} header .decoration {display:none!important} </style>", unsafe_allow_html=True)


model = load_model("./detection.h5")


model.summary()
st.title("Upload Image")

data = st.file_uploader("", type=["png", "jpg", "jpeg"])
if data:
    st.title("Detected Image:")
    img_src = PIL.Image.open(data)
    img_data = img_src.resize((299, 299))
    img = img_to_array(img_data)
    img_array = preprocess_input(np.expand_dims(img, axis=0))
    heatmap = make_gradcam_heatmap(img_array, model)

   # overlay the heatmap and image
    heatmap = np.uint8(255 * heatmap)  # heatmap to rgb (0-255)

    # we create an image from the heatmap
    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = array_to_img(jet_heatmap)
    # matching original image size
    jet_heatmap = jet_heatmap.resize((299, 299))
    jet_heatmap = img_to_array(jet_heatmap)

    # we overlay the two images
    # the heatmap is weighted so that we can see the image through it
    # the heatmap is weighted so that we can see the image through it
    superimposed_img = jet_heatmap * 0.003 + img
    superimposed_img = array_to_img(superimposed_img)  # create new image
    superimposed_img.resize((299, 299))

    st.image(superimposed_img)
