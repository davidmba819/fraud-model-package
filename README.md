# Fraud Detection ML Package

An end-to-end machine learning package for fraud detection using XGBoost.

## Features

- Data loading
- Data preprocessing
- Model training
- Model evaluation
- Fraud prediction
- Threshold-based classification

## Project Structure

```text
fraud_model/
├── data/
├── fraud_model/
├── model/
├── requirements.txt
├── setup.py
├── MANIFEST.in
└── README.md
```

## Installation

```bash
pip install -e .
```

## Train

```bash
python -m fraud_model.train_pipeline
```

## Evaluate

```bash
python -m fraud_model.evaluate
```

## Predict

```python
from fraud_model.predict import predict_model
```