import uvicorn
from fastapi import FastAPI, File, UploadFile
from joblib import load
from scipy.sparse import load_npz
import joblib
import io



app = FastAPI()

classifier = joblib.load("xgb_model.pkl")

@app.get('/')
def index():
    return {'message': 'hello, world'}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    npz_file = io.BytesIO(contents)


    x_test_merged_loaded = load_npz(npz_file)
    predictions = classifier.predict(x_test_merged_loaded)

    return{
        'predictions': predictions.tolist()
    }