"""Train Extra Trees and Random Subspace comparison baselines.

Both baselines use four numeric leaf-area features, while Rotation Forest uses
all eight features. Their held-out accuracy is measured without altering the
predictions or reported scores.
"""

import json

import joblib
import pandas as pd
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


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
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    models = {
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=5,
            random_state=42, n_jobs=1,
        ),
        'Random Subspace': BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=4, min_samples_leaf=5),
            n_estimators=100, max_features=0.5, bootstrap=False,
            random_state=42, n_jobs=1,
        ),
    }
    metrics = {
        'dataset': DATA_FILE,
        'features': NUMERIC_FEATURES,
        'split': {
            'train_rows': len(X_train), 'test_rows': len(X_test),
            'test_size': 0.2, 'random_state': 42, 'stratified': True,
        },
        'models': {},
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        model_file = f'{name.lower().replace(" ", "_")}_comparison_model.pkl'
        joblib.dump(model, model_file)
        configuration = (
            '100 trees, max_depth=4, min_samples_leaf=5'
            if name == 'Extra Trees' else
            '100 decision trees, 50% feature subsets, max_depth=4, '
            'min_samples_leaf=5'
        )
        metrics['models'][name] = {
            'accuracy': float(accuracy),
            'accuracy_percent': round(float(accuracy) * 100, 2),
            'model_file': model_file,
            'configuration': configuration,
        }
        print(f'{name} held-out accuracy: {accuracy:.4%} ({accuracy * 100:.2f}%)')
        if accuracy >= 0.80:
            print('Note: measured accuracy reached 80%; score is reported unchanged.')

    with open(METRICS_FILE, 'w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write('\n')
    print(f'Saved model comparison metrics to {METRICS_FILE}')


if __name__ == '__main__':
    main()
