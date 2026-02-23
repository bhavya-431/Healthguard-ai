from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.inference import MedicalChatbot
import os

app = FastAPI()

# Global chatbot instance
chatbot = None

@app.on_event("startup")
async def startup_event():
    global chatbot
    print("Loading Chatbot Model...")
    chatbot = MedicalChatbot()
    print("Chatbot Loaded.")

class SymptomRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_disease(request: SymptomRequest):
    if chatbot is None:
        return {"error": "Model not loaded properly"}
    
    predictions = chatbot.predict(request.text)
    if isinstance(predictions, str): # Handle error messages
        return {"message": predictions, "predictions": []}
        
    return {
        "predictions": predictions,
        "message": f"Based on your symptoms, the most likely condition is {predictions[0]['disease']}."
    }

# Mount static files
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, 'index.html'))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

