import numpy as np

def feature_fusion(facial_features, speech_features, body_features):
    # Simple concatenation of features
    return np.concatenate((facial_features.flatten(), speech_features.flatten(), body_features.flatten()))
