# Credit Card Fraud Detection

This project is Task 5 of the CodSoft Data Science Internship. It builds a machine learning classification model to predict whether a credit card transaction is genuine or fraudulent.

## Dataset

Dataset used: Credit Card Fraud Detection dataset  
Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the dataset file in the project folder as:

```text
creditcard.csv
```

## Project Files

- `train_model.py` - loads the dataset, performs basic EDA, preprocesses data, trains the model, evaluates it, and saves the trained pipeline.
- `app.py` - Streamlit web application for checking whether a transaction is genuine or fraudulent.
- `requirements.txt` - required Python packages.
- `fraud_detection_pipeline.pkl` - saved trained model and preprocessing pipeline.
- `feature_metadata.json` - saved feature information used by the Streamlit app.

## Features

- Displays dataset shape, column names, missing values, and class counts.
- Identifies `Class` as the target column.
- Handles class imbalance using balanced class weights.
- Uses stratified train-test split.
- Scales numerical features using `StandardScaler`.
- Trains a Logistic Regression classification model.
- Evaluates the model using accuracy, precision, recall, and F1-score.
- Provides a Streamlit interface for entering transaction feature values.
- Displays prediction result and fraud probability.

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train_model.py
```

The training script prints:

```python
print(f"Model Accuracy: {accuracy:.2%}")
```

It also prints precision, recall, and F1-score.

## Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Prediction Output

The app displays one of the following results:

- `Genuine Transaction`
- `Fraudulent Transaction`

If probability prediction is available, it also displays the fraud probability.
