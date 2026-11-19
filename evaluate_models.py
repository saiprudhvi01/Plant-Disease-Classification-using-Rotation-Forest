"""Evaluate the saved Rotation Forest, SVM, and KNN models side by side."""

import os

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT, 'plant_disease_symptom_observations_18000.csv')
FEATURE_COLUMNS = [
    'Crop',
    'Symptom_Name',
    'Symptom_Status',
    'Coverage_Percent',
    'Severity',
    'Lesion_Area_Percent',
    'Yellowing_Area_Percent',
    'Necrosis_Area_Percent',
]
NUMERIC_COLUMNS = [
    'Coverage_Percent',
    'Lesion_Area_Percent',
    'Yellowing_Area_Percent',
    'Necrosis_Area_Percent',
]
CATEGORICAL_COLUMNS = ['Crop', 'Symptom_Name', 'Symptom_Status', 'Severity']


def main():
    print('Loading saved models and evaluation data...')
    data = pd.read_csv(DATA_FILE)
    label_encoders = joblib.load(os.path.join(ROOT, 'label_encoders.pkl'))
    target_encoder = joblib.load(os.path.join(ROOT, 'target_encoder.pkl'))
    rotation_scaler = joblib.load(os.path.join(ROOT, 'scaler.pkl'))
    rotation_forest = joblib.load(os.path.join(ROOT, 'rotation_forest_model.pkl'))
    svm = joblib.load(os.path.join(ROOT, 'svm_comparison_model.pkl'))
    knn = joblib.load(os.path.join(ROOT, 'knn_comparison_model.pkl'))

    X = data[FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_COLUMNS:
        X[column] = label_encoders[column].transform(X[column].astype(str))
    y = target_encoder.transform(data['Disease'].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    del X_train, y_train

    rotation_predictions = rotation_forest.predict(rotation_scaler.transform(X_test))
    svm_predictions = svm.predict(X_test[NUMERIC_COLUMNS])
    knn_predictions = knn.predict(X_test[NUMERIC_COLUMNS])
    predictions = {
        'Rotation Forest (8 features)': rotation_predictions,
        'SVM baseline (4 numeric features)': svm_predictions,
        'KNN baseline (4 numeric features)': knn_predictions,
    }

    disease_names = target_encoder.classes_
    print(f'\nDataset: {os.path.basename(DATA_FILE)} ({len(data):,} rows)')
    print(f'Test set: {len(y_test):,} rows; stratified 80/20 split, random_state=42')
    print('Macro metrics weight each disease class equally; weighted metrics account for class support.')

    for model_name, predicted in predictions.items():
        print('\n' + '=' * 88)
        print(model_name)
        print('-' * 88)
        print(f'Accuracy:          {accuracy_score(y_test, predicted) * 100:7.2f}%')
        print(f'Precision (macro): {precision_score(y_test, predicted, average="macro", zero_division=0) * 100:7.2f}%')
        print(f'Recall (macro):    {recall_score(y_test, predicted, average="macro", zero_division=0) * 100:7.2f}%')
        print(f'F1 (macro):        {f1_score(y_test, predicted, average="macro", zero_division=0) * 100:7.2f}%')
        print(f'Precision (weighted): {precision_score(y_test, predicted, average="weighted", zero_division=0) * 100:7.2f}%')
        print(f'Recall (weighted):    {recall_score(y_test, predicted, average="weighted", zero_division=0) * 100:7.2f}%')
        print(f'F1 (weighted):        {f1_score(y_test, predicted, average="weighted", zero_division=0) * 100:7.2f}%')

        matrix = confusion_matrix(y_test, predicted, labels=range(len(disease_names)))
        matrix_frame = pd.DataFrame(matrix, index=disease_names, columns=disease_names)
        matrix_frame.index.name = 'Actual \\ Predicted'
        print('\nConfusion matrix (rows = actual, columns = predicted):')
        print(matrix_frame.to_string())

    print('\nComparison note: SVM and KNN intentionally use only four numeric leaf-area features;')
    print('Rotation Forest uses all eight features, so this is not a feature-parity comparison.')
    print('These are same-dataset holdout metrics, not cross-dataset validation results.')


if __name__ == '__main__':
    main()
