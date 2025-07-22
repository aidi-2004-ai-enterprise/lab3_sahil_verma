# Project Name
Sahil Lab3

## Description
AIDI 2004
This individual assignment focuses on building a machine learning pipeline using the Seaborn penguins dataset, training an XGBoost model, and deploying it via a FastAPI application. You will preprocess the data, train and evaluate a model, and create a prediction endpoint with proper input validation. The assignment emphasizes robust error handling, logging, and professional coding practices.

## Installation

uv run train.py

uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Then use it with swagger: http://127.0.0.1:8000/docs





