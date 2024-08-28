import cv2
import dlib
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Load and preprocess the FER2013 dataset
def load_fer2013_data(data_dir):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

# Build a CNN model for facial emotion recognition
def build_emotion_model(input_shape):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Train the model
def train_emotion_model(model, train_data, validation_data, epochs=25):
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        verbose=1
    )
    
    return history

# Paths to dataset
data_dir = 'path_to_fer2013_data_directory'

train_data, validation_data = load_fer2013_data(data_dir)
emotion_model = build_emotion_model((48, 48, 1))
train_emotion_model(emotion_model, train_data, validation_data)

# Save the trained model
emotion_model.save('emotion_model.h5')

'using FER 2013 DATASET'
