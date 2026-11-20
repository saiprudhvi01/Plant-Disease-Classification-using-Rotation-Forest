# Plant Disease Prediction System

A Flask web app that predicts plant diseases from crop and leaf symptom observations. Users select a crop, choose the symptoms that are present, enter the affected area for each symptom, and receive a ranked disease prediction. The app also stores prediction history locally and can export individual or complete history reports as CSV.

## Technology stack

| Area | Technologies |
| --- | --- |
| Web server | Python, Flask |
| User interface | HTML, CSS, JavaScript, Jinja templates, inline SVG |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, custom Rotation Forest implementation |
| Model persistence | joblib and pickle artifacts |
| Prediction history | SQLite |
| Deployment runtime | Render native Python runtime, pinned with `.python-version` |

Dependency versions are listed in [`requirements.txt`](requirements.txt). The project pins Python to the 3.10 minor version so the installed scientific packages remain compatible with the saved model artifacts.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Flask routes, model inference, symptom lookup, history storage, and CSV report downloads |
| `templates/base.html` | Shared page layout, navigation, theme toggle, and styles |
| `templates/index.html` | Prediction page with crop-specific symptom selection and results |
| `templates/history.html` | Saved prediction history and CSV report links |
| `static/css/fluttertop.css` | Shared Phytotech hero and shell styling |
| `static/images/plant.svg`, `static/images/leaf-circle.webp` | Hero plant illustration and leaf thumbnail |
| `rotation_forest.py` | Custom Rotation Forest classifier used by the main model |
| `retrain_model.py` | Main Rotation Forest training and evaluation script |
| `rotation_forest_model.py` | Earlier, self-contained training experiment; not needed to run the app |
| `train_comparison_models.py` | Trains the Extra Trees and Random Subspace comparison baselines |
| `evaluate_models.py` | Generates model-comparison and confusion-matrix graphs and prints evaluation metrics |
| `plant_disease_symptom_observations_18000.csv` | Main labeled training and symptom lookup dataset |
| `rotation_forest_model.pkl` | Saved Rotation Forest model loaded by the app |
| `scaler.pkl` | Saved standardization transform for the Rotation Forest features |
| `label_encoders.pkl` | Encoders for categorical input features |
| `target_encoder.pkl` | Encoder for disease labels |
| `extra_trees_comparison_model.pkl`, `random_subspace_comparison_model.pkl` | Saved comparison models |
| `comparison_models_metrics.json` | Holdout metrics and training metadata displayed in the UI |

The local SQLite database (`prediction_history.db`) is created automatically and is ignored by Git. The large `crop_disease_remediation.csv` and generated `predictions_with_rotation_forest.csv` are not required by the web app and are also ignored.

## Dataset and prediction features

The main dataset contains 18,000 observations, 12 disease labels, and 6 crops. Each row has eight input features and a `Disease` target:

1. `Crop`
2. `Symptom_Name`
3. `Symptom_Status`
4. `Coverage_Percent`
5. `Severity`
6. `Lesion_Area_Percent`
7. `Yellowing_Area_Percent`
8. `Necrosis_Area_Percent`

The crop selector filters the symptom choices using crop/symptom pairs from this dataset. During inference, the app builds one model input for each selected symptom with a positive affected-area percentage. It uses a matching crop/symptom row as a template for severity and leaf-area fields, replaces the coverage percentage with the user's value, predicts each symptom row, then averages the probability vectors to rank the final diseases. If every selected symptom has 0% affected area, the app applies an explicit rule-based `Healthy` result and skips model inference; no model confidence is assigned to that result. Zero-area symptoms are omitted from inference when other selected symptoms have positive coverage.

Model probabilities are not artificially raised. The displayed confidence is the highest averaged probability after restricting and renormalizing to the selected crop's disease classes. If it is below 60%, the result is marked for manual review in the results panel, saved history, and CSV report. This is a raw model probability threshold, not a separately calibrated confidence estimate. For the all-zero Healthy rule, no model probability is calculated and confidence is shown as N/A.

The classifier is trained across all 12 disease classes, then inference restricts and renormalizes the final ranking to the diseases associated with the selected crop. For example, Apple results can only be Apple Scab or Cedar Apple Rust. The reported confidence is conditional on the selected crop's disease classes, not across every crop in the dataset.

## Main model training steps

Run the primary training script from the repository root:

```bash
python retrain_model.py
```

The script performs these steps:

1. Reads `plant_disease_symptom_observations_18000.csv`.
2. Separates the eight input features from the `Disease` target.
3. Label-encodes `Crop`, `Symptom_Name`, `Symptom_Status`, and `Severity`; label-encodes the target labels separately.
4. Makes a stratified 80/20 train/test split using `random_state=42` (14,400 training rows and 3,600 test rows for the current dataset).
5. Fits a `StandardScaler` on the training feature rows and applies it to both splits.
6. Fits a 100-estimator Rotation Forest. Each estimator uses a random subset of four features, applies PCA to that subset, and trains a decision tree with `max_depth=15`.
7. Reports training and holdout accuracy, then writes the model and preprocessing artifacts to the `.pkl` files listed above.

The saved Rotation Forest was evaluated on the same-dataset 3,600-row holdout: **3,589 correct, 99.69% accuracy**. This is an in-dataset random holdout score, not a cross-dataset validation result. The custom implementation uses random feature subsets without a fixed NumPy seed, so a newly trained model can produce a slightly different score.

## Extra Trees and Random Subspace comparison models

Run:

```bash
python train_comparison_models.py
```

The trained files are saved in the project root (the same folder as `app.py`): `extra_trees_comparison_model.pkl` and `random_subspace_comparison_model.pkl`. They are already present there after training. The former SVM and KNN model files have been removed.

These are intentionally **limited-feature baselines**: they use only the four numeric leaf-area measurements (`Coverage_Percent`, `Lesion_Area_Percent`, `Yellowing_Area_Percent`, and `Necrosis_Area_Percent`). They do not use crop or symptom category features, while the Rotation Forest uses all eight input features. Both are configured with shallow trees to keep the comparison models simpler; measured scores are reported as-is.

The script uses the same stratified 80/20 split (`random_state=42`), fits both classifiers on the training rows, evaluates them on the holdout, then saves the models and `comparison_models_metrics.json`.

| Model | Configuration | Held-out accuracy |
| --- | --- | ---: |
| Extra Trees | 100 trees, max depth 4 | 54.03% |
| Random Subspace | 100 trees, 50% feature subsets, max depth 4 | 50.81% |

These scores are from the current dataset and split. They are not cross-dataset scores. The metrics JSON is generated by the training script; rerun the script after changing the training data or configurations.

To print an aligned side-by-side metrics table and create separate CSV files for all three 12-class confusion matrices, run:

```bash
python evaluate_models.py
```

The script evaluates the saved model artifacts on the deterministic stratified 20% holdout and generates a model-metrics bar chart plus one confusion-matrix heatmap per model. It also prints a compact metrics table and saves the underlying numbers as CSV files. Confusion matrices use numeric class indices rather than disease names. Extra Trees and Random Subspace use four numeric features; Rotation Forest uses all eight features. Graphs and CSV files are written under `evaluation_reports/`.

## Install and run locally

Use Python 3.10 and run:

```bash
python -m venv .venv
```

Activate the virtual environment, then install dependencies and start the app:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/` in a browser. The app reads model artifacts and the main CSV from the repository root. Keep those files in place when running locally.

## Web routes

| Route | Purpose |
| --- | --- |
| `GET /` | Serves the prediction UI |
| `GET /history-page` | Serves the saved-predictions history page |
| `POST /predict` | Predicts a disease from the selected crop and symptoms; saves successful predictions to SQLite |
| `GET /history` | Returns the 25 most recent saved predictions as JSON |
| `GET /history/report.csv` | Downloads all saved prediction history as CSV |
| `GET /history/<prediction_id>/report.csv` | Downloads one prediction as CSV |
| `POST /batch_predict` | Accepts a CSV upload containing the model's feature columns and returns batch predictions |

Downloaded prediction reports contain the date, crop, selected symptoms and affected-area percentages, predicted disease, confidence, top-three disease names, prediction method, and manual-review status/reason. Rule-based Healthy results show confidence as N/A.

## Render deployment

The root `.python-version` file pins Render to Python 3.10, which is compatible with the versions in `requirements.txt` and the saved model artifacts. For a native Python web service, use:

- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`

The Flask entry point binds to the `PORT` environment variable supplied by Render, falling back to port 5000 for local development.
