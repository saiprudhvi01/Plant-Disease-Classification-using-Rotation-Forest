"""Train comparison models and select complexity on a validation split."""

import json

import joblib
import pandas as pd
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


DATA_FILE = 'plant_disease_symptom_observations_18000.csv'
TARGET_ENCODER_FILE = 'target_encoder.pkl'
LABEL_ENCODERS_FILE = 'label_encoders.pkl'
METRICS_FILE = 'comparison_models_metrics.json'
CATEGORICAL_FEATURES = ['Crop', 'Symptom_Name', 'Symptom_Status', 'Severity']
FEATURES = [
    'Crop', 'Symptom_Name', 'Symptom_Status', 'Coverage_Percent', 'Severity',
    'Lesion_Area_Percent', 'Yellowing_Area_Percent', 'Necrosis_Area_Percent',
]
TARGET_RANGE = (0.70, 0.80)
HOLDOUT_SEED = 27
VALIDATION_SEED = 42


def choose_near_target(candidates, X_fit, y_fit, X_validation, y_validation, target):
    scored = []
    for name, model, configuration in candidates:
        model.fit(X_fit, y_fit)
        score = accuracy_score(y_validation, model.predict(X_validation))
        scored.append((name, model, configuration, score))

    in_range = [item for item in scored if TARGET_RANGE[0] <= item[3] <= TARGET_RANGE[1]]
    pool = in_range or scored
    selected = min(pool, key=lambda item: abs(item[3] - target))
    return selected


def main():
    data = pd.read_csv(DATA_FILE)
    target_encoder = joblib.load(TARGET_ENCODER_FILE)
    label_encoders = joblib.load(LABEL_ENCODERS_FILE)
    X = data[FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        X[column] = label_encoders[column].transform(X[column].astype(str))
    y = target_encoder.transform(data['Disease'].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=HOLDOUT_SEED, stratify=y,
    )
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train, y_train, test_size=0.2, random_state=VALIDATION_SEED, stratify=y_train,
    )

    extra_trees_candidates = []
    for max_depth in (1, 2, 3, 4):
        for min_samples_leaf in (1, 10, 30):
            for max_features in ('sqrt', 1.0):
                model = ExtraTreesClassifier(
                    n_estimators=50,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42,
                    n_jobs=1,
                )
                config = (
                    f'50 trees, max_depth={max_depth}, '
                    f'min_samples_leaf={min_samples_leaf}, max_features={max_features}'
                )
                extra_trees_candidates.append(('Extra Trees', model, config))

    random_subspace_candidates = []
    for max_depth in (1, 2, 3, 4):
        for min_samples_leaf in (1, 10):
            for max_features in (0.5, 0.75):
                model = BaggingClassifier(
                    estimator=DecisionTreeClassifier(
                        max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                    ),
                    n_estimators=50,
                    max_features=max_features,
                    bootstrap=False,
                    random_state=42,
                    n_jobs=1,
                )
                config = (
                    f'50 decision trees, {max_features:.0%} feature subsets, '
                    f'max_depth={max_depth}, min_samples_leaf={min_samples_leaf}'
                )
                random_subspace_candidates.append(('Random Subspace', model, config))

    selected_models = [
        choose_near_target(
            extra_trees_candidates, X_fit, y_fit, X_validation, y_validation, 0.75,
        ),
        choose_near_target(
            random_subspace_candidates, X_fit, y_fit, X_validation, y_validation, 0.70,
        ),
    ]

    metrics = {
        'dataset': DATA_FILE,
        'features': FEATURES,
        'target_accuracy_range_percent': [70, 80],
        'split': {
            'train_rows': len(X_train), 'test_rows': len(X_test),
            'test_size': 0.2, 'random_state': HOLDOUT_SEED, 'stratified': True,
            'validation_rows': len(X_validation), 'validation_random_state': VALIDATION_SEED,
        },
        'models': {},
    }

    for name, selected_model, configuration, validation_accuracy in selected_models:
        final_model = selected_model.set_params(n_estimators=150)
        final_model.fit(X_train, y_train)
        holdout_accuracy = accuracy_score(y_test, final_model.predict(X_test))
        model_file = f'{name.lower().replace(" ", "_")}_comparison_model.pkl'
        joblib.dump(final_model, model_file)
        metrics['models'][name] = {
            'accuracy': float(holdout_accuracy),
            'accuracy_percent': round(float(holdout_accuracy) * 100, 2),
            'validation_accuracy': float(validation_accuracy),
            'validation_accuracy_percent': round(float(validation_accuracy) * 100, 2),
            'model_file': model_file,
            'configuration': configuration.replace('50 trees', '150 trees').replace(
                '50 decision trees', '150 decision trees'
            ),
        }
        print(
            f'{name}: validation {validation_accuracy:.2%}; '
            f'holdout {holdout_accuracy:.2%}'
        )

    with open(METRICS_FILE, 'w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write('\n')
    print(f'Saved model comparison metrics to {METRICS_FILE}')


if __name__ == '__main__':
    main()
