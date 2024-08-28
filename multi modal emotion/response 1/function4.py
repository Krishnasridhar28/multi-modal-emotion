from keras.models import load_model
from utils.preprocessing import preprocess_facial_expression
from utils.datasets import load_facial_expression_data

model = load_model('models/emotion_model.h5')
detector = dlib.get_frontal_face_detector()

def recognize_facial_expression(image_path):
    image = load_facial_expression_data(image_path)
    preprocessed_image = preprocess_facial_expression(image)
    predictions = model.predict(preprocessed_image)
    return predictions

'''face recoginition'''
