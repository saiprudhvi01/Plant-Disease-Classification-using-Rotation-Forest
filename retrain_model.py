import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')
from rotation_forest import RotationForest

# Load data
print("Loading data...")
df = pd.read_csv('plant_disease_symptom_observations_18000.csv')
print(f"Dataset shape: {df.shape}")

# Separate features and target
X = df.drop('Disease', axis=1)
y = df['Disease']

# Encode categorical features
print("Encoding categorical features...")
categorical_cols = ['Crop', 'Symptom_Name', 'Symptom_Status', 'Severity']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Rotation Forest
print("Training Rotation Forest...")
rf = RotationForest(
    n_estimators=100,
    n_features_subset=4,
    max_depth=15
)
rf.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = rf.predict(X_train_scaled)
y_pred_test = rf.predict(X_test_scaled)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Save model and encoders
print("\nSaving model...")
joblib.dump(rf, 'rotation_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

print("Model saved successfully!")
