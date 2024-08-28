import cv2
from openpose import pyopenpose as op
from utils.preprocessing import preprocess_body_keypoints

params = {"model_folder": "models/pose_model/"}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def interpret_body_language(image_path):
    frame = cv2.imread(image_path)
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    keypoints = preprocess_body_keypoints(datum.poseKeypoints)
    return keypoints, datum.cvOutputData
' body reconginition'
