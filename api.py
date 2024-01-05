
from fastapi import FastAPI, Depends, Header, HTTPException,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from pyod_ad import inference_model_result
import uvicorn
from logging_handler import Logger
import time


app = FastAPI()

class Summary(BaseModel):
    metadata: Dict
    sender: str
    persona_id: str
    persona_config:Dict
    message: str
    time_period:str


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/AnomalyDetect')
async def create_place_view(input: Summary):

    requestId = input.sender
    use_cache = True

    start_time = time.time()
    Logger.info("Received a request {0} for Anomaly Deep-dive service with cache = {1}".format(requestId, use_cache))
    kpi_cols = [ kpi_dict['col_name']  for kpi_dict in input.persona_config["persona_metrics"]]
    result = inference_model_result(requestId, kpi_cols, use_cache)
    end_time = time.time()
    Logger.info("request {} for Anomaly Deep-dive service with cache = {} took {:.4f} seconds".format(requestId, use_cache, end_time - start_time))
    

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9704)
