import streamlit as st
import numpy as np
import os
from streamlit_drawable_canvas import st_canvas

import cnn


# Load some values
CANVAS_SIZE = 250
RATIO_STROKE = 7
EXAMPLES = 25

# Show title and subtitle
st.title("MNIST Prediction with a CNN")
st.markdown("A CNN model is used to predict the category of images in the Mnist dataset. Please draw a digit in the canvas and look at the prediction made !") 

# Build a sidebar
with st.sidebar:
    st.write("""
        Configuration of the CNN.
    """)

    epochs = st.slider("Epochs", min_value=1, max_value=50, value=cnn.EPOCHS)
    batch = st.slider("Batch size", min_value=16, max_value=256, value=cnn.BATCH_SIZE)
    shear = st.slider("Shear ratio", min_value=0.0, max_value=1.0, value=cnn.SHEAR_RATIO)
    shift = st.slider("Shift ratio", min_value=0.0, max_value=1.0, value=cnn.SHIFT_RATIO)
    zoom = st.slider("Zoom ratio", min_value=0.0, max_value=2.0, value=cnn.ZOOM_RATIO)
    rotation = st.slider("Rotation angle", min_value=0, max_value=90, value=cnn.ROTATION_ANGLE)
    augmented = st.checkbox('Augmented dataset')

    # If clicked, build the model
    if st.button("Construire le modèle"):
        cnn.build(
            epochs=epochs, 
            batch_size=batch, 
            augmented=augmented, 
            rotation=rotation, 
            zoom=zoom, 
            shift=shift, 
            shear=shear,
        )

# Two columns
col_draw, col_process = st.columns(2)

# The canvas
with col_draw:
    canvas = st_canvas(
    stroke_width=CANVAS_SIZE/RATIO_STROKE,
    height=CANVAS_SIZE,
    width=CANVAS_SIZE,
)

# Load the dataframe of random images
df = cnn.load_random_dataset()

# If things are drawn
if canvas.image_data is not None:

    # Load the model
    model = cnn.load(augmented=augmented)

    # Convert the canvas image to a image for the model
    image_model = cnn.resize(canvas.image_data)

    # Convert it back to the canvas size and show it
    image_processed = cnn.resize(image_model, size=(CANVAS_SIZE, CANVAS_SIZE))

    with col_process:
        st.image(image_processed)


    # Make the prediction
    predictions, val = cnn.predict(model, image_model[:,:,3])
    
    # If the prediction seems to be ok    
    if np.any(predictions > 0.75):
        st.subheader(f'Other pictures of a {val}')

        # Get images of the same digit
        df_images = df.loc[df.y == val].sample(frac=1).head(EXAMPLES).x.to_numpy()
        
        # Show those iamges
        images = st.columns(EXAMPLES)
        for image, df_image in zip(images, df_images):

            with image:
                st.image(
                    df_image.reshape(cnn.IMG_SIZE, cnn.IMG_SIZE),
                    width=35
                )

