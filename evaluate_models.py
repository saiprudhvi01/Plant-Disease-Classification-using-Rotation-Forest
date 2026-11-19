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
REPORT_DIR = os.path.join(ROOT, 'evaluation_reports')
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
        'Rotation Forest (8 features)': ('rotation_forest', rotation_predictions),
        'SVM baseline (4 numeric features)': ('svm_baseline', svm_predictions),
        'KNN baseline (4 numeric features)': ('knn_baseline', knn_predictions),
    }

    disease_names = target_encoder.classes_
    print(f'\nDataset: {os.path.basename(DATA_FILE)} ({len(data):,} rows)')
    print(f'Test set: {len(y_test):,} rows; stratified 80/20 split, random_state=42')
    print('Macro scores weight every disease equally; weighted scores account for class size.')

    os.makedirs(REPORT_DIR, exist_ok=True)
    metric_rows = []
    for model_name, (file_prefix, predicted) in predictions.items():
        metric_rows.append({
            'Model': model_name,
            'Accuracy (%)': accuracy_score(y_test, predicted) * 100,
            'Precision macro (%)': precision_score(y_test, predicted, average='macro', zero_division=0) * 100,
            'Recall macro (%)': recall_score(y_test, predicted, average='macro', zero_division=0) * 100,
            'F1 macro (%)': f1_score(y_test, predicted, average='macro', zero_division=0) * 100,
            'Precision weighted (%)': precision_score(y_test, predicted, average='weighted', zero_division=0) * 100,
            'Recall weighted (%)': recall_score(y_test, predicted, average='weighted', zero_division=0) * 100,
            'F1 weighted (%)': f1_score(y_test, predicted, average='weighted', zero_division=0) * 100,
        })
        matrix = confusion_matrix(y_test, predicted, labels=range(len(disease_names)))
        matrix_frame = pd.DataFrame(matrix, index=disease_names, columns=disease_names)
        matrix_frame.index.name = 'Actual disease'
        matrix_frame.to_csv(os.path.join(REPORT_DIR, f'{file_prefix}_confusion_matrix.csv'))

    summary = pd.DataFrame(metric_rows)
    summary.to_csv(os.path.join(REPORT_DIR, 'model_metrics_summary.csv'), index=False, float_format='%.2f')

    columns = [
        ('Model', 38, None),
        ('Accuracy %', 12, 'Accuracy (%)'),
        ('Precision %', 12, 'Precision macro (%)'),
        ('Recall %', 12, 'Recall macro (%)'),
        ('F1 %', 12, 'F1 macro (%)'),
    ]
    print('\nMODEL METRICS (MACRO AVERAGE)')
    print(' | '.join(title.ljust(width) for title, width, _ in columns))
    print('-+-'.join('-' * width for _, width, _ in columns))
    for row in metric_rows:
        values = [row['Model']] + [f"{row[key]:.2f}" for _, _, key in columns[1:]]
        print(' | '.join(value.ljust(width) if index == 0 else value.rjust(width)
                          for index, (value, (_, width, _)) in enumerate(zip(values, columns)))
              )

    print('\nCONFUSION MATRICES')
    print(f'Full matrices saved as separate CSV files in: {REPORT_DIR}')
    print(f'Macro and weighted metrics saved to: {os.path.join(REPORT_DIR, "model_metrics_summary.csv")}')
    print('Rows are actual diseases; columns are predicted diseases.')
    print('\nComparison note: SVM and KNN use four numeric features; Rotation Forest uses all eight.')
    print('These are same-dataset holdout metrics, not cross-dataset validation results.')


if __name__ == '__main__':
    main()
