import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

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
