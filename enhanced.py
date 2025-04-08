#!/usr/bin/env python3
"""
Production-level Real-Time Emotion Analysis Tool

This tool captures video from a camera and uses facial emotion analysis (via DeepFace),
body language detection (via MediaPipe Pose), and speech sentiment analysis to generate
insights for counsellors. It includes an educational mode that prints explanations,
real-time alerts for prolonged negative cues, and a post-session survey for feedback.

Usage:
    python emotion_analysis.py --duration 120 --educational True

Dependencies:
    - OpenCV (cv2)
    - mediapipe
    - deepface
    - speechrecognition
    - numpy
    - argparse
    - logging
    - json

Note: This tool is for demonstration purposes and should be further tested and secured
before deployment in a production environment.
"""

import cv2
import mediapipe as mp
from deepface import DeepFace
import speech_recognition as sr
import time
import json
import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("emotion_analysis.log")
    ]
)

class EmotionAnalysisSystem:
    def __init__(self, session_duration=120, educational_mode=True, negative_threshold=3, speech_interval=10):
        """
        Initialize the Emotion Analysis System.

        Args:
            session_duration (int): Minimum session duration in seconds.
            educational_mode (bool): Flag to enable educational messages.
            negative_threshold (int): Number of consecutive negative cues before alert.
            speech_interval (int): Seconds between each speech analysis.
        """
        self.session_duration = session_duration
        self.educational_mode = educational_mode
        self.negative_threshold = negative_threshold
        self.speech_interval = speech_interval

        # Initialize MediaPipe Pose model
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()

        # Initialize insights log and negativity counter
        self.insight_log = []
        self.consecutive_negative_count = 0

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.error("Cannot access the camera.")
            raise RuntimeError("Camera access failed.")

        logging.info("Emotion Analysis System initialized.")

    def provide_psychological_insights(self, face_emotion, text_emotion, body_movement):
        """Return insights based on the detected cues."""
        insights = []
        if face_emotion == "sad":
            insights.append("User appears sad. Provide supportive or uplifting responses.")
        elif face_emotion == "happy":
            insights.append("User appears happy. Encourage them to share positive experiences.")
        elif face_emotion == "neutral":
            insights.append("User appears neutral. Ask open-ended questions to gather more information.")
        else:
            insights.append("No strong facial emotion detected.")

        if text_emotion == "negative":
            insights.append("Speech tone indicates negativity. Consider creating a safe space for discussion.")
        elif text_emotion == "positive":
            insights.append("Speech tone indicates positivity. Encourage continued positive dialogue.")

        if body_movement == "slouched":
            insights.append("Body language indicates slouching, suggesting low energy or disengagement.")
        elif body_movement == "upright":
            insights.append("Body language indicates attentiveness and confidence.")

        return "\n".join(insights)

    @staticmethod
    def explain_results(entry):
        """Return detailed explanations of the observations (for educational mode)."""
        explanation = []
        explanation.append("Explanation of Observations:")
        explanation.append(f"1. Face Emotion ('{entry['face_emotion']}'): Derived from facial expression analysis. 'Sad' may indicate the need for support.")
        explanation.append(f"2. Speech Emotion ('{entry['speech_emotion']}'): Derived from speech analysis. 'Negative' suggests distress, while 'positive' indicates an upbeat tone.")
        explanation.append(f"3. Body Language ('{entry['body_language']}'): 'Upright' usually suggests engagement; 'slouched' can suggest low energy.")
        explanation.append("These insights are intended to assist counsellors, not replace professional judgment.")
        return "\n".join(explanation)

    def detect_emotion_from_face(self, frame):
        """Detect and return the dominant facial emotion using DeepFace."""
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list) and analysis:
                return analysis[0].get('dominant_emotion', 'neutral')
            elif isinstance(analysis, dict):
                return analysis.get('dominant_emotion', 'neutral')
        except Exception as e:
            logging.error(f"Facial emotion detection error: {e}")
        return "neutral"

    def detect_body_language(self, frame):
        """Detect and return the body posture (upright or slouched) using MediaPipe Pose."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            if results.pose_landmarks:
                left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                if abs(left_shoulder.y - right_shoulder.y) > 0.1:
                    return "slouched"
                return "upright"
        except Exception as e:
            logging.error(f"Body language detection error: {e}")
        return "unknown"

    def get_speech_emotion(self):
        """Capture speech, convert to text, and analyze sentiment."""
        with sr.Microphone() as source:
            logging.info("Listening for speech...")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                logging.info(f"Speech-to-text: {text}")
                if any(word in text.lower() for word in ["sad", "bad", "angry", "depressed"]):
                    return "negative", text
                elif any(word in text.lower() for word in ["happy", "good", "great", "excited"]):
                    return "positive", text
                return "neutral", text
            except (sr.WaitTimeoutError, sr.UnknownValueError) as e:
                logging.warning("Speech recognition timed out or could not understand.")
                return "neutral", ""

    def run(self):
        """Run the main analysis loop for the specified session duration."""
        logging.info("Starting session.")
        session_start_time = time.time()
        last_speech_time = time.time()
        speech_emotion, speech_text = "neutral", ""

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("Failed to read frame from camera.")
                    break

                # Analyze facial emotion and body language
                face_emotion = self.detect_emotion_from_face(frame)
                body_language = self.detect_body_language(frame)

                # Check for speech emotion periodically
                if time.time() - last_speech_time > self.speech_interval:
                    speech_emotion, speech_text = self.get_speech_emotion()
                    last_speech_time = time.time()

                # Generate insights for this frame
                insights = self.provide_psychological_insights(face_emotion, speech_emotion, body_language)
                current_entry = {
                    "timestamp": time.time(),
                    "face_emotion": face_emotion,
                    "speech_emotion": speech_emotion,
                    "body_language": body_language,
                    "speech_text": speech_text,
                    "insights": insights
                }
                self.insight_log.append(current_entry)

                # Check for prolonged negative cues
                if face_emotion == "sad" or speech_emotion == "negative":
                    self.consecutive_negative_count += 1
                else:
                    self.consecutive_negative_count = 0

                alert_message = ""
                if self.consecutive_negative_count >= self.negative_threshold:
                    alert_message = "Alert: Prolonged negative indicators detected. Consider intervention!"
                    cv2.putText(frame, alert_message, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    logging.warning(alert_message)

                # Overlay analysis results on the video feed
                cv2.putText(frame, f"Face: {face_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Speech: {speech_emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Body: {body_language}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Press 'q' to quit (after session duration)", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("Real-Time Emotion Analysis", frame)

                if self.educational_mode:
                    logging.info("Educational Tip: 'sad' face may require support; 'upright' posture indicates engagement.")

                # Check for exit key (only allow exit after minimum session duration)
                key = cv2.waitKey(1) & 0xFF
                elapsed_time = time.time() - session_start_time
                if key == ord('q') and elapsed_time >= self.session_duration:
                    logging.info("User requested exit after minimum session duration.")
                    break
                if elapsed_time >= self.session_duration:
                    logging.info("Minimum session duration reached. Exiting session.")
                    break

        except Exception as e:
            logging.exception(f"An error occurred during the session: {e}")
        finally:
            # Clean up resources
            self.cap.release()
            cv2.destroyAllWindows()
            self.save_session_log()
            logging.info("Session ended. Remember: This tool is an aid and should complement professional judgment.")

    def save_session_log(self, filename="insight_log.json"):
        """Save the session log as a JSON file."""
        try:
            with open(filename, "w") as f:
                json.dump(self.insight_log, f, indent=2)
            logging.info(f"Session log saved to {filename}.")
        except Exception as e:
            logging.error(f"Failed to save session log: {e}")


def run_survey():
    """Conduct a post-session survey and save responses to a JSON file."""
    print("\n--- Post-Session Feedback Survey ---")
    responses = {}
    responses["overall_experience"] = input("How would you rate your overall experience? (1-5): ")
    responses["ease_of_use"] = input("How easy was it to use the tool? (1-5): ")
    responses["usefulness"] = input("How useful did you find the insights provided? (1-5): ")
    responses["comments"] = input("Any additional comments or suggestions: ")

    try:
        with open("survey_responses.json", "a") as f:
            f.write(json.dumps(responses) + "\n")
        print("Thank you for your feedback!")
    except Exception as e:
        print(f"Error saving survey responses: {e}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Real-Time Emotion Analysis Tool")
    parser.add_argument("--duration", type=int, default=120, help="Minimum session duration in seconds")
    parser.add_argument("--educational", type=lambda x: x.lower() == 'true', default=True, help="Enable educational mode (True/False)")
    return parser.parse_args()


def main():
    # Print disclaimer and require consent
    print("Disclaimer: This tool is designed to assist counsellors by providing supplementary insights from video, audio, and biometric data. It is not a substitute for professional judgment.")
    consent = input("Do you consent to the collection and analysis of video, audio, and biometric data? (y/n): ")
    if consent.lower() != "y":
        print("Consent not given. Exiting the program.")
        return

    args = parse_args()
    try:
        system = EmotionAnalysisSystem(
            session_duration=args.duration,
            educational_mode=args.educational
        )
        system.run()
    except Exception as e:
        logging.exception(f"Fatal error: {e}")

    # Run post-session survey
    run_survey()


if __name__ == "__main__":
    main()
