from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import os
from .utils import load_model

app = FastAPI(
    title="News Category Classifier API",
    description="API for classifying news headlines into categories",
    version="1.0.0"
)

# Load the model and label encoder
try:
    model, label_encoder = load_model('models')
    print("Model and label encoder loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {
        "message": "Welcome to News Category Classifier API",
        "endpoints": {
            "predict": "/predict (POST) - Classify news headline",
            "docs": "/docs - Interactive API documentation"
        }
    }

@app.post("/predict")
async def predict_category(request: TextRequest):
    try:
        # Preprocess the input text
        processed_text = preprocess_text(request.text)
        
        # Make prediction
        prediction = model.predict([processed_text])
        
        # Convert numeric prediction to category name
        category = label_encoder.inverse_transform(prediction)[0]
        
        return {
            "text": request.text,
            "processed_text": processed_text,
            "predicted_category": category,
            "category_id": int(prediction[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)