import cv2
from openpose import pyopenpose as op

# Set up OpenPose parameters
params = {
    "model_folder": "path_to_openpose_models/",
}

# Initialize OpenPose with the correct parameters
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def interpret_body_language(frame):
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    return datum.cvOutputData, datum.poseKeypoints

# Example to capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    output_frame, keypoints = interpret_body_language(frame)
    cv2.imshow("Body Language Interpretation", output_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'using coco datasets'
