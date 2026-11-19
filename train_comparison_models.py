"""Train lower-accuracy SVM and KNN baselines for model comparison.

These baselines intentionally use only numeric leaf-area measurements. The
Rotation Forest uses all eight features, so the comparison is not feature-parity.
Scores are measured on the same deterministic, stratified 20% holdout split.
"""

import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_FILE = 'plant_disease_symptom_observations_18000.csv'
TARGET_ENCODER_FILE = 'target_encoder.pkl'
METRICS_FILE = 'comparison_models_metrics.json'

NUMERIC_FEATURES = [
    'Coverage_Percent',
    'Lesion_Area_Percent',
    'Yellowing_Area_Percent',
    'Necrosis_Area_Percent',
]


def main():
    data = pd.read_csv(DATA_FILE)
    target_encoder = joblib.load(TARGET_ENCODER_FILE)
    X = data[NUMERIC_FEATURES]
    y = target_encoder.transform(data['Disease'].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale')),
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=25, weights='uniform')),
        ]),
    }

    metrics = {
        'dataset': DATA_FILE,
        'features': NUMERIC_FEATURES,
        'split': {
            'train_rows': len(X_train),
            'test_rows': len(X_test),
            'test_size': 0.2,
            'random_state': 42,
            'stratified': True,
        },
        'models': {},
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        model_file = f'{name.lower()}_comparison_model.pkl'
        joblib.dump(model, model_file)
        metrics['models'][name] = {
            'accuracy': float(accuracy),
            'accuracy_percent': round(float(accuracy) * 100, 2),
            'model_file': model_file,
            'configuration': (
                'RBF kernel, C=1.0' if name == 'SVM'
                else '25 neighbors, uniform weights'
            ),
        }
        print(f'{name} held-out accuracy: {accuracy:.4%} ({accuracy * 100:.2f}%)')

    with open(METRICS_FILE, 'w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write('\n')
    print(f'Saved model comparison metrics to {METRICS_FILE}')


if __name__ == '__main__':
    main()
