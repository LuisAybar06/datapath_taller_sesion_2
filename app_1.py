import uvicorn
from fastapi import FastAPI
from joblib import load
from scipy.sparse import load_npz
import joblib



app = FastAPI()


@app.get('/')
def index():
    return {'message': 'hello, world'}

@app.post('/predict')
def predict():
    classifier = joblib.load("xgb_model.pkl")
    x_test_merge_load = load_npz("x_test_merged.npz")

    predictions = classifier.predict(x_test_merge_load)

    return{
        'predictions': predictions.tolist()
    }