import os
import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load and preprocess the RAVDESS dataset
def load_ravdess_data(data_dir):
    emotions = []
    features = []
    
    for file in os.listdir(data_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(data_dir, file)
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            features.append(mfccs_scaled)
            
            emotion = file.split('-')[2]  # Extract emotion label from file name
            emotions.append(emotion)
    
    return np.array(features), np.array(emotions)

# Encode the emotions and split the dataset
def prepare_ravdess_data(features, emotions):
    le = LabelEncoder()
    emotions_encoded = le.fit_transform(emotions)
    emotions_one_hot = np.eye(len(le.classes_))[emotions_encoded]
    
    X_train, X_test, y_train, y_test = train_test_split(features, emotions_one_hot, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, le

# Build a simple neural network model for speech emotion recognition
def build_speech_model(input_shape):
    model = Sequential()
    
    model.add(Dense(256, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(8, activation='softmax'))  # 8 emotions in RAVDESS
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Paths to dataset
data_dir = 'path_to_ravdess_data_directory'

features, emotions = load_ravdess_data(data_dir)
X_train, X_test, y_train, y_test, le = prepare_ravdess_data(features, emotions)
speech_model = build_speech_model(X_train.shape[1])

# Train the model
speech_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
speech_model.save('speech_emotion_model.h5')

'RAVDESS DATASET'
