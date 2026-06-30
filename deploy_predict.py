import pickle
from deploy_features import extract_all_features_deploy, ONNXMobileNet

_cnn_model = None
_meta_bundle = None

def _load_models():
    global _cnn_model, _meta_bundle
    if _cnn_model is not None:
        return
    
    _cnn_model = ONNXMobileNet("mobilenet.onnx")
    
    with open("meta_classifier.pkl", "rb") as f:
        _meta_bundle = pickle.load(f)

def predict(image_path: str) -> float:
    _load_models()
    
    features = extract_all_features_deploy(image_path, _cnn_model)
    features_2d = features.reshape(1, -1)
    
    clf = _meta_bundle['model']
    scaler = _meta_bundle['scaler']
    
    features_scaled = scaler.transform(features_2d)
    prob = clf.predict_proba(features_scaled)[0, 1]
    
    return float(prob)
