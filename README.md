# heartDiseaseProjectML

# Heart Disease Machine Learning Classification

This project develops a machine learning pipeline for early prediction of heart disease using clinical patient data. The task is binary classification: predicting heart disease presence or absence.

## Models Used

- Logistic Regression
- Support Vector Machine
- Random Forest

## Data Split

The dataset was split into:

- 60% training data
- 10% evaluation/validation data for hyperparameter tuning
- 30% final testing data

## Files

- `clean_heart_data.py`: Cleans the raw UCI heart disease datasets, handles missing values, encodes categorical variables, normalizes numeric features, and exports the cleaned Excel file.
- `train_heart_models.py`: Trains Logistic Regression, SVM, and Random Forest models. Performs grid search hyperparameter tuning and evaluates the best models on the final test set.
- `data/`: Contains raw and processed datasets.
- `outputs/`: Contains model results, tuning tables, charts, and confusion matrices.

## How to Run

First, install dependencies:

```bash
pip install -r requirements.txt
