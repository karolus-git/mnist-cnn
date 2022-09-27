import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from pathlib import Path

# Load some constants
BATCH_SIZE = 32
EPOCHS = 5
SHIFT_RATIO = 0.2
ZOOM_RATIO = 0.2
SHEAR_RATIO = 0.2
ROTATION_ANGLE = 15
RANDOM_SEED = 42
IMG_SIZE = 28
NUM_CLASSES = 10


# Load the dataframe of random pictures
def load_random_dataset():
    path = Path("./mnist-cnn/models/random_dataset.pkl")

    if not path.exists():
        load_data()

    return pd.read_pickle(path)
    
# Create the CNN model
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPool2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Save a dataframe of digits
def save_random_data(x,y):

    df = pd.DataFrame.from_dict({
        "x" : list(x.reshape(60000, IMG_SIZE*IMG_SIZE)),
        "y" : y
    })

    df.sample(frac=1).to_pickle("./mnist-cnn/models/random_dataset.pkl")


def resize(img, size=(IMG_SIZE, IMG_SIZE)):
    return cv2.resize(img, size,  interpolation=cv2.INTER_NEAREST)

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Save digits to dataframe
    save_random_data(x_train, y_train)

    return (x_train, y_train), (x_test, y_test)

def build(
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        augmented=False, 
        shear=SHEAR_RATIO, 
        zoom=ZOOM_RATIO, 
        rotation=ROTATION_ANGLE,
        shift=SHIFT_RATIO
    ):

    
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = load_data()
   
    # Reshape the vectors
    x_train = x_train.reshape(60000,IMG_SIZE, IMG_SIZE,1) / 255.0
    y_train = to_categorical(y_train, NUM_CLASSES)

    x_test = x_test.reshape(10000,IMG_SIZE,IMG_SIZE,1) / 255.0
    y_test = to_categorical(y_test, NUM_CLASSES)

    name = "model.h5" if not augmented else "model_aug.h5"

    # Create the model
    model = create_model()

    if not augmented:

        # Fit the model
        model.fit(
            x_train, y_train,
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(x_test, y_test))
    else:
        # Build an image generator
        dataGen = ImageDataGenerator(
            rotation_range=rotation,
            width_shift_range=shift,
            height_shift_range=shift,
            SHEAR_RATIO=shear,
            ZOOM_RATIO=[0.1,1.1],
            validation_split=0.2)

        # Fit the image generator
        dataGen.fit(x_train)

        train_generator = dataGen.flow(x_train, y_train, 
            batch_size=batch_size, 
            shuffle=True, 
            seed=RANDOM_SEED, 
            save_to_dir=None, 
            subset='training')

        validation_generator = dataGen.flow(x_train, y_train, 
            batch_size=batch_size, 
            shuffle=True, 
            seed=RANDOM_SEED, 
            save_to_dir=None, 
            subset='validation')

        # Fit the model
        model.fit_generator(train_generator,
            epochs=epochs,
            validation_data = validation_generator)

    # Save the model
    model.save(f'./mnist-cnn/models/{name}')

    





def load(augmented=False):
    return models.load_model('./mnist-cnn/models/model_aug.h5') if augmented else models.load_model('./mnist-cnn/models/model.h5')

def predict(model, x):

    x = x.reshape(1, IMG_SIZE, IMG_SIZE)

    predictions = model.predict(x)

    return predictions, np.argmax(predictions)



