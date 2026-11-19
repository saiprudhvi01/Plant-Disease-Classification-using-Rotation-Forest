from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime, timezone
from io import BytesIO, StringIO
import csv
import json
import sqlite3
import os
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from rotation_forest import RotationForest

app = Flask(__name__)
HISTORY_DB = os.path.join(app.root_path, 'prediction_history.db')


def init_history_db():
    with sqlite3.connect(HISTORY_DB) as connection:
        connection.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                crop TEXT NOT NULL,
                symptoms_json TEXT NOT NULL,
                predicted_disease TEXT NOT NULL,
                top_predictions_json TEXT NOT NULL
            )
        ''')


init_history_db()

# Load model and encoders
print("Loading model and encoders...")
model = joblib.load('rotation_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

# Get unique values for dropdowns
crop_options = list(label_encoders['Crop'].classes_)
symptom_options = list(label_encoders['Symptom_Name'].classes_)
symptom_data = pd.read_csv('plant_disease_symptom_observations_18000.csv', usecols=['Crop', 'Symptom_Name'])
crop_symptom_map = {
    crop: sorted(group['Symptom_Name'].dropna().astype(str).unique().tolist())
    for crop, group in symptom_data.groupby('Crop')
}
symptom_status_options = list(label_encoders['Symptom_Status'].classes_)
severity_options = list(label_encoders['Severity'].classes_)
disease_classes = list(target_encoder.classes_)
with open(os.path.join(app.root_path, 'comparison_models_metrics.json'), encoding='utf-8') as metrics_file:
    comparison_metrics = json.load(metrics_file)

@app.route('/')
def home():
    return render_template('index.html', 
                         crop_options=crop_options,
                         symptom_options=symptom_options,
                         crop_symptom_map=crop_symptom_map,
                         symptom_status_options=symptom_status_options,
                         severity_options=severity_options,
                         disease_classes=disease_classes,
                         comparison_metrics=comparison_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        crop = data['crop']
        symptoms = data['symptoms']
        
        if not symptoms or len(symptoms) == 0:
            return jsonify({'success': False, 'error': 'No symptoms selected'})
        
        # Load original data to find matching patterns for each symptom
        df_original = pd.read_csv('plant_disease_symptom_observations_18000.csv')
        
        # Collect predictions for each symptom
        all_predictions = []
        all_probabilities = []
        
        for symptom_data in symptoms:
            symptom_name = symptom_data['name']
            symptom_percentage = symptom_data['percentage']
            
            # Find matching rows in original data for this crop and symptom
            matching_rows = df_original[
                (df_original['Crop'] == crop) & 
                (df_original['Symptom_Name'] == symptom_name)
            ]
            
            if len(matching_rows) == 0:
                # If no exact match, try to find rows with just the symptom
                matching_rows = df_original[df_original['Symptom_Name'] == symptom_name]
            
            if len(matching_rows) == 0:
                continue
            
            # Use the first matching row as template, but override with user's percentage
            template_row = matching_rows.iloc[0].copy()
            
            # Create feature dictionary with user's percentage
            features = {
                'Crop': crop,
                'Symptom_Name': symptom_name,
                'Symptom_Status': 'Present',
                'Coverage_Percent': symptom_percentage,
                'Severity': template_row['Severity'],
                'Lesion_Area_Percent': template_row['Lesion_Area_Percent'],
                'Yellowing_Area_Percent': template_row['Yellowing_Area_Percent'],
                'Necrosis_Area_Percent': template_row['Necrosis_Area_Percent']
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([features])
            
            # Encode categorical features
            for col in ['Crop', 'Symptom_Name', 'Symptom_Status', 'Severity']:
                input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
            
            # Scale features
            input_scaled = scaler.transform(input_df)
            
            # Make prediction for this symptom
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            all_predictions.append(prediction)
            all_probabilities.append(probability)
        
        if not all_predictions:
            return jsonify({'success': False, 'error': 'Could not match symptoms to training data'})
        
        # Average the probabilities from all symptoms
        avg_probability = np.mean(all_probabilities, axis=0)
        final_prediction = np.argmax(avg_probability)
        confidence = np.max(avg_probability)
        
        # Decode prediction
        predicted_disease = target_encoder.inverse_transform([final_prediction])[0]
        
        # Get top 3 predictions with probabilities
        top_3_indices = np.argsort(avg_probability)[-3:][::-1]
        top_3_predictions = []
        for idx in top_3_indices:
            top_3_predictions.append({
                'disease': target_encoder.inverse_transform([idx])[0],
                'probability': float(avg_probability[idx])
            })
        
        result = {
            'success': True,
            'predicted_disease': predicted_disease,
            'confidence': float(confidence),
            'top_3_predictions': top_3_predictions,
            'symptoms_analyzed': len(symptoms)
        }

        created_at = datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')
        with sqlite3.connect(HISTORY_DB) as connection:
            cursor = connection.execute(
                '''INSERT INTO prediction_history
                   (created_at, crop, symptoms_json, predicted_disease, top_predictions_json)
                   VALUES (?, ?, ?, ?, ?)''',
                (created_at, crop, json.dumps(symptoms), predicted_disease,
                 json.dumps([prediction['disease'] for prediction in top_3_predictions]))
            )
            result['prediction_id'] = cursor.lastrowid
            result['created_at'] = created_at
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


def history_rows(limit=None):
    query = '''SELECT id, created_at, crop, symptoms_json, predicted_disease, top_predictions_json
               FROM prediction_history ORDER BY id DESC'''
    params = ()
    if limit is not None:
        query += ' LIMIT ?'
        params = (limit,)
    with sqlite3.connect(HISTORY_DB) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(query, params).fetchall()]


def csv_report(records):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Prediction ID', 'Date', 'Crop', 'Selected Symptoms', 'Predicted Disease', 'Top 3 Diseases'])
    for record in records:
        symptoms = json.loads(record['symptoms_json'])
        top_predictions = json.loads(record['top_predictions_json'])
        symptom_text = '; '.join(
            f"{item['name']} ({item.get('percentage', 0)}%)" for item in symptoms
        )
        writer.writerow([
            record['id'], record['created_at'], record['crop'], symptom_text,
            record['predicted_disease'], '; '.join(top_predictions)
        ])
    return BytesIO(('\ufeff' + output.getvalue()).encode('utf-8'))


@app.route('/history', methods=['GET'])
def get_history():
    records = history_rows(limit=25)
    return jsonify({
        'success': True,
        'history': [{
            'id': record['id'],
            'created_at': record['created_at'],
            'crop': record['crop'],
            'symptoms': json.loads(record['symptoms_json']),
            'predicted_disease': record['predicted_disease'],
            'top_predictions': json.loads(record['top_predictions_json'])
        } for record in records]
    })


@app.route('/history/report.csv', methods=['GET'])
def download_history_report():
    report = csv_report(history_rows())
    return send_file(report, mimetype='text/csv', as_attachment=True,
                     download_name='plant_disease_prediction_history.csv')


@app.route('/history/<int:prediction_id>/report.csv', methods=['GET'])
def download_prediction_report(prediction_id):
    record = next((row for row in history_rows() if row['id'] == prediction_id), None)
    if record is None:
        return jsonify({'success': False, 'error': 'Prediction not found'}), 404
    report = csv_report([record])
    return send_file(report, mimetype='text/csv', as_attachment=True,
                     download_name=f'plant_disease_prediction_{prediction_id}.csv')

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Make sure required columns exist
        required_cols = ['Crop', 'Symptom_Name', 'Symptom_Status', 'Coverage_Percent', 
                        'Severity', 'Lesion_Area_Percent', 'Yellowing_Area_Percent', 'Necrosis_Area_Percent']
        
        for col in required_cols:
            if col not in df.columns:
                return jsonify({'success': False, 'error': f'Missing column: {col}'})
        
        # Encode categorical features
        for col in ['Crop', 'Symptom_Name', 'Symptom_Status', 'Severity']:
            df[col] = label_encoders[col].transform(df[col].astype(str))
        
        # Scale features
        X = df[required_cols]
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        confidences = np.max(probabilities, axis=1)
        
        # Decode predictions
        predicted_diseases = target_encoder.inverse_transform(predictions)
        
        # Add predictions to dataframe
        result_df = pd.read_csv(file)
        result_df['Predicted_Disease'] = predicted_diseases
        result_df['Confidence'] = confidences
        
        # Convert to JSON
        results = result_df.to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'results': results,
            'total_predictions': len(results)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', '5000')))
