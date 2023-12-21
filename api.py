
from fastapi import FastAPI, Depends, Header, HTTPException,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from pyod_ad import inference_model_result
import uvicorn
from logging_handler import Logger

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/AnomalyDetect')
async def create_place_view():
    
    return inference_model_result()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9702)