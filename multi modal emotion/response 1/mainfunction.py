import cv2
from facial_expression_recognition import recognize_facial_expression
from speech_emotion_recognition import recognize_speech_emotion
from body_language_interpretation import interpret_body_language
from utils.fusion import feature_fusion

def run_multi_modal_emotion_recognition(image_path, audio_path):
    # Recognize facial expression
    facial_features = recognize_facial_expression(image_path)

    # Recognize speech emotion
    speech_features = recognize_speech_emotion(audio_path)

    # Interpret body language
    body_features, output_frame = interpret_body_language(image_path)

    # Feature Fusion
    fused_features = feature_fusion(facial_features, speech_features, body_features)
    
    # Example output
    print(f"Fused Features: {fused_features}")
    cv2.imshow("Body Language Interpretation", output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

