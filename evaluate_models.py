"""Evaluate Rotation Forest, Extra Trees, and Random Subspace models."""

import os

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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


def save_metric_chart(metric_rows):
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    metric_keys = ['Accuracy (%)', 'Precision macro (%)', 'Recall macro (%)', 'F1 macro (%)']
    x_positions = np.arange(len(metric_names))
    width = 0.23
    colors = ['#15803D', '#2563EB', '#D97706']
    legend_handles = []

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model_index, row in enumerate(metric_rows):
        values = [row[key] for key in metric_keys]
        bars = ax.bar(
            x_positions + (model_index - 1) * width,
            values,
            width,
            label=row['Model'],
            color=colors[model_index],
            edgecolor='white',
            linewidth=0.8,
        )
        legend_handles.append(bars)
        ax.bar_label(bars, fmt='%.1f', padding=2, fontsize=8)

    fig.suptitle('Model Comparison on the Same Test Set', fontsize=14, weight='bold', y=0.99)
    ax.set_ylabel('Score (%)')
    ax.set_xticks(x_positions, metric_names)
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    fig.legend(
        handles=legend_handles,
        labels=[row['Model'] for row in metric_rows],
        frameon=False,
        ncol=3,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.93),
        fontsize=8,
    )
    fig.text(
        0.5, 0.01,
        'Extra Trees and Random Subspace use four numeric features; Rotation Forest uses all eight.',
        ha='center', fontsize=8, color='#555555'
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.84))
    output_path = os.path.join(REPORT_DIR, 'model_metrics_comparison.png')
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return output_path


def save_confusion_matrix_chart(matrix, title, file_prefix):
    class_indices = np.arange(1, matrix.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(12, 9))
    image = ax.imshow(matrix, cmap='Greens', interpolation='nearest')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label='Number of samples')
    ax.set_title(f'{title}\nConfusion Matrix', fontsize=14, weight='bold', pad=14)
    ax.set_xlabel('Predicted class index')
    ax.set_ylabel('Actual class index')
    ax.set_xticks(np.arange(len(class_indices)), class_indices, fontsize=8)
    ax.set_yticks(np.arange(len(class_indices)), class_indices, fontsize=8)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            count = int(matrix[row_index, column_index])
            if count:
                ax.text(
                    column_index, row_index, str(count),
                    ha='center', va='center',
                    color='white' if count > threshold else '#17301F',
                    fontsize=8,
                )

    fig.tight_layout()
    output_path = os.path.join(REPORT_DIR, f'{file_prefix}_confusion_matrix.png')
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return output_path


def main():
    print('Loading saved models and evaluation data...')
    data = pd.read_csv(DATA_FILE)
    label_encoders = joblib.load(os.path.join(ROOT, 'label_encoders.pkl'))
    target_encoder = joblib.load(os.path.join(ROOT, 'target_encoder.pkl'))
    rotation_scaler = joblib.load(os.path.join(ROOT, 'scaler.pkl'))
    rotation_forest = joblib.load(os.path.join(ROOT, 'rotation_forest_model.pkl'))
    extra_trees = joblib.load(os.path.join(ROOT, 'extra_trees_comparison_model.pkl'))
    random_subspace = joblib.load(os.path.join(ROOT, 'random_subspace_comparison_model.pkl'))

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
    extra_trees_predictions = extra_trees.predict(X_test[NUMERIC_COLUMNS])
    random_subspace_predictions = random_subspace.predict(X_test[NUMERIC_COLUMNS])
    predictions = {
        'Rotation Forest (8 features)': ('rotation_forest', rotation_predictions),
        'Extra Trees (4 numeric features)': ('extra_trees', extra_trees_predictions),
        'Random Subspace (4 numeric features)': ('random_subspace', random_subspace_predictions),
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
        class_indices = np.arange(len(disease_names))
        matrix = confusion_matrix(y_test, predicted, labels=class_indices)
        display_indices = np.arange(1, len(disease_names) + 1)
        matrix_frame = pd.DataFrame(matrix, index=display_indices, columns=display_indices)
        matrix_frame.index.name = 'Actual class index'
        matrix_frame.to_csv(os.path.join(REPORT_DIR, f'{file_prefix}_confusion_matrix.csv'))
        save_confusion_matrix_chart(matrix, model_name, file_prefix)

    summary = pd.DataFrame(metric_rows)
    summary.to_csv(os.path.join(REPORT_DIR, 'model_metrics_summary.csv'), index=False, float_format='%.2f')
    metric_chart_path = save_metric_chart(metric_rows)

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
    print(f'Comparison chart: {metric_chart_path}')
    print(f'Confusion-matrix graphs and CSVs: {REPORT_DIR}')
    print(f'Macro and weighted metrics saved to: {os.path.join(REPORT_DIR, "model_metrics_summary.csv")}')
    print('Confusion matrices use numeric class indices without disease names.')
    print('\nComparison note: Extra Trees and Random Subspace use four numeric features; Rotation Forest uses all eight.')
    print('These are same-dataset holdout metrics, not cross-dataset validation results.')


if __name__ == '__main__':
    main()
