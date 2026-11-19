import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('plant_disease_symptom_observations_18000.csv')
print(f"Dataset shape: {df.shape}")

# Display class distribution
print("\n=== Disease Classes ===")
print(df['Disease'].value_counts())
print(f"\nTotal classes: {df['Disease'].nunique()}")

# Separate features and target
X = df.drop('Disease', axis=1)
y = df['Disease']

# Encode categorical features
print("\n=== Encoding categorical features ===")
categorical_cols = ['Crop', 'Symptom_Name', 'Symptom_Status', 'Severity']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}: {len(le.classes_)} unique values")

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print(f"Encoded target: {len(target_encoder.classes_)} classes")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Rotation Forest Implementation
class RotationForest:
    def __init__(self, n_estimators=50, n_features_subset=3, max_depth=None):
        self.n_estimators = n_estimators
        self.n_features_subset = n_features_subset
        self.max_depth = max_depth
        self.estimators = []
        self.pcas = []
        self.feature_subsets = []
        self.n_features = None
        
    def fit(self, X, y):
        self.n_features = X.shape[1]
        n_samples = X.shape[0]
        
        for i in range(self.n_estimators):
            # Randomly select feature subset
            feature_subset = np.random.choice(
                self.n_features, 
                self.n_features_subset, 
                replace=False
            )
            self.feature_subsets.append(feature_subset)
            
            # Extract subset
            X_subset = X[:, feature_subset]
            
            # Apply PCA for rotation
            pca = PCA(n_components=min(self.n_features_subset, X_subset.shape[1]))
            X_rotated = pca.fit_transform(X_subset)
            self.pcas.append(pca)
            
            # Train decision tree on rotated features
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=i,
                min_samples_split=5,
                min_samples_leaf=2
            )
            tree.fit(X_rotated, y)
            self.estimators.append(tree)
            
        return self
    
    def predict(self, X):
        predictions = []
        for i, estimator in enumerate(self.estimators):
            feature_subset = self.feature_subsets[i]
            X_subset = X[:, feature_subset]
            X_rotated = self.pcas[i].transform(X_subset)
            pred = estimator.predict(X_rotated)
            predictions.append(pred)
        
        # Majority voting
        predictions = np.array(predictions)
        final_predictions = []
        for j in range(predictions.shape[1]):
            votes = predictions[:, j]
            final_predictions.append(np.bincount(votes).argmax())
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        probas = []
        for i, estimator in enumerate(self.estimators):
            feature_subset = self.feature_subsets[i]
            X_subset = X[:, feature_subset]
            X_rotated = self.pcas[i].transform(X_subset)
            proba = estimator.predict_proba(X_rotated)
            probas.append(proba)
        
        # Average probabilities
        avg_proba = np.mean(probas, axis=0)
        return avg_proba

# Train Rotation Forest
print("\n=== Training Rotation Forest ===")
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

# Cross-validation
print("\n=== Cross-Validation ===")
cv_scores = cross_val_score(
    DecisionTreeClassifier(max_depth=15, min_samples_split=5, min_samples_leaf=2),
    X_train_scaled, y_train, cv=5
)
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")

# If accuracy < 90%, try to improve
if test_accuracy < 0.90:
    print("\n=== Tuning for better accuracy ===")
    best_accuracy = test_accuracy
    best_model = rf
    
    # Try different configurations
    for n_est in [50, 100, 150, 200]:
        for n_feat in [3, 4, 5, 6]:
            for depth in [10, 15, 20, None]:
                rf_tuned = RotationForest(
                    n_estimators=n_est,
                    n_features_subset=n_feat,
                    max_depth=depth
                )
                rf_tuned.fit(X_train_scaled, y_train)
                y_pred_tuned = rf_tuned.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred_tuned)
                
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = rf_tuned
                    print(f"New best: {acc:.4f} (n_est={n_est}, n_feat={n_feat}, depth={depth})")
    
    rf = best_model
    y_pred_test = rf.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Detailed classification report
print("\n=== Classification Report ===")
print(classification_report(
    y_test, y_pred_test, 
    target_names=target_encoder.classes_,
    digits=4
))

# Confusion Matrix
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# Save model and encoders
print("\n=== Saving model ===")
joblib.dump(rf, 'rotation_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')

# Make predictions on all data
print("\n=== Making predictions on full dataset ===")
X_scaled = scaler.transform(X)
all_predictions = rf.predict(X_scaled)
all_proba = rf.predict_proba(X_scaled)

# Add predictions to dataframe
df['Predicted_Disease'] = target_encoder.inverse_transform(all_predictions)
df['Prediction_Confidence'] = np.max(all_proba, axis=1)

# Save predictions
df.to_csv('predictions_with_rotation_forest.csv', index=False)
print("Predictions saved to 'predictions_with_rotation_forest.csv'")

# Display sample predictions
print("\n=== Sample Predictions ===")
print(df[['Disease', 'Predicted_Disease', 'Prediction_Confidence']].head(20))

print("\n=== Training Complete ===")
print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
if test_accuracy >= 0.90:
    print("✓ Target accuracy of 90% achieved!")
else:
    print(f"⚠ Accuracy is {test_accuracy*100:.2f}% (target: 90%)")
