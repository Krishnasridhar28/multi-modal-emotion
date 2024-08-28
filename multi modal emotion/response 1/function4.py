import numpy as np
from keras.models import load_model
from utils.preprocessing import preprocess_speech
from utils.datasets import load_speech_data

model = load_model('models/speech_model.h5')

def recognize_speech_emotion(audio_path):
    mfccs = load_speech_data(audio_path)
    preprocessed_mfccs = preprocess_speech(mfccs)
    predictions = model.predict(preprocessed_mfccs)
    return predictions

'voice recoginition'
