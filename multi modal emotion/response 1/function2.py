
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def preprocess_facial_expression(image):
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image / 255.0

def preprocess_speech(mfccs):
    mfccs = np.expand_dims(mfccs, axis=0)
    return scaler.transform(mfccs)

def preprocess_body_keypoints(keypoints):
    return keypoints / np.linalg.norm(keypoints)
