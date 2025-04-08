import cv2
import mediapipe as mp
from deepface import DeepFace
import speech_recognition as sr

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to generate psychological insights
def provide_psychological_insights(face_emotion, text_emotion, body_movement):
    insights = []

    if face_emotion == "sad":
        insights.append("The user seems sad. Provide supportive or uplifting responses.")
    elif face_emotion == "happy":
        insights.append("The user seems happy. Encourage them to share positive experiences.")
    elif face_emotion == "neutral":
        insights.append("The user appears neutral. Ask open-ended questions to understand their state.")
    else:
        insights.append("No strong emotion detected from the face.")

    # Insights based on speech emotion
    if text_emotion == "negative":
        insights.append("The speech tone suggests negativity. Create a safe space for expressing concerns.")
    elif text_emotion == "positive":
        insights.append("The speech tone suggests positivity. Encourage maintaining this outlook.")

    # Insights based on body language
    if body_movement == "slouched":
        insights.append("Body language indicates slouching. This could suggest low energy or disengagement.")
    elif body_movement == "upright":
        insights.append("Body language indicates attentiveness and confidence.")

    return "\n".join(insights)

# Real-time emotion detection from the face
def detect_emotion_from_face(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(analysis, list) and len(analysis) > 0:
            return analysis[0].get('dominant_emotion', 'neutral')
        elif isinstance(analysis, dict):
            return analysis.get('dominant_emotion', 'neutral')
        else:
            return "neutral"
    except Exception as e:
        print(f"Error in facial emotion detection: {e}")
        return "neutral"

# Real-time body movement detection
def detect_body_language(frame):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            # Analyze body posture
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            if abs(left_shoulder.y - right_shoulder.y) > 0.1:  # Slouching detection
                return "slouched"
            else:
                return "upright"
        return "unknown"
    except Exception as e:
        print(f"Error in body language detection: {e}")
        return "unknown"

# Real-time speech-to-text
def get_speech_emotion():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for speech...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"Speech-to-text: {text}")
            # Simple sentiment analysis (keyword matching for demo purposes)
            if any(neg in text.lower() for neg in ["sad", "bad", "angry", "depressed"]):
                return "negative", text
            elif any(pos in text.lower() for pos in ["happy", "good", "great", "excited"]):
                return "positive", text
            else:
                return "neutral", text
        except sr.WaitTimeoutError:
            print("Speech recognition timed out.")
            return "neutral", ""
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return "neutral", ""

# Main function
def main():
    cap = cv2.VideoCapture(0)
    print("Initializing system... Speak and look into the camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect face emotion
        face_emotion = detect_emotion_from_face(frame)

        # Detect body language
        body_language = detect_body_language(frame)

        # Get speech-to-text and sentiment
        text_emotion, speech_text = get_speech_emotion()

        # Generate psychological insights
        insights = provide_psychological_insights(face_emotion, text_emotion, body_language)
        print("\nPsychological Insights:")
        print(insights)

        # Display the video feed with detected information
        cv2.putText(frame, f"Face Emotion: {face_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Body Language: {body_language}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('Real-Time Analysis', frame)

        # Exit the program by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    main()
