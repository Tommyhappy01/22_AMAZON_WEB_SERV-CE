#!/usr/bin/env python3
import uvicorn
import sklearn
import pickle
import joblib
import numpy as np
import pandas as pd
from preprocessing import cleaning
from fastapi import FastAPI, Request, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates # web template engine for python.
tfidf_vec = open('models/tfidf_vectorizer.pkl', 'rb')
tfidf_vec = joblib.load(tfidf_vec)
model = open('models/logreg_tweet_classifier.pkl', 'rb')
model = joblib.load(model)
# init app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
# ML Aspect Sample 1
@app.get("/predict/{tweet}")
async def predict(tweet: str):
    cleaned_text = pd.Series(tweet).apply(cleaning)
    vectorized_text = tfidf_vec.transform([tweet]).toarray()
    prediction = model.predict(vectorized_text)
    prob = model.predict_proba(vectorized_text)
    return {"original_text":tweet, "prediction":prediction[0], "Prob":np.max(np.round(prob,3))}
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("item.html", {"request": request})
# POST end-point
@app.post("/get_tweet")
async def handle_tweet(request: Request, tweet: str = Form(...)):
    cleaned_text = pd.Series(tweet).apply(cleaning)
    vectorized_text = tfidf_vec.transform([tweet]).toarray()
    prediction = model.predict(vectorized_text)
    prob = model.predict_proba(vectorized_text)
    return templates.TemplateResponse("item.html",
            {"request": request, "original_text":tweet, "prediction":prediction[0],
             "Prob":np.max(np.round(prob,3))})
if __name__=='__main__':
    uvicorn.run(app, port=5000)