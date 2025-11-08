"""
Machine learning classifiers for BCI applications
"""

from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class BCIClassifier:
    """BCI classifier with multiple algorithms and evaluation metrics."""
    
    def __init__(self, algorithm: str = 'svm', random_state: int = 42):
        """
        Initialize BCI classifier.
        
        Parameters
        ----------
        algorithm : str
            Classification algorithm ('svm', 'rf', 'lda', 'lr', 'knn')
        random_state : int
            Random state for reproducibility
        """
        self.algorithm = algorithm
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._create_model()
    
    def _create_model(self):
        """Create the specified model."""
        if self.algorithm == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=self.random_state,
                probability=True
            )
        elif self.algorithm == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.algorithm == 'lda':
            self.model = LinearDiscriminantAnalysis(
                solver='svd'
            )
        elif self.algorithm == 'lr':
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=-1
            )
        elif self.algorithm == 'knn':
            self.model = KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, normalize: bool = True):
        """
        Fit the classifier.
        
        Parameters
        ----------
        X : np.ndarray
            Training features (n_samples, n_features)
        y : np.ndarray
            Training labels (n_samples,)
        normalize : bool
            Whether to normalize features
        """
        if normalize:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
            
        self.model.fit(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Test features (n_samples, n_features)
        normalize : bool
            Whether to normalize features
            
        Returns
        -------
        predictions : np.ndarray
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if normalize:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Test features (n_samples, n_features)
        normalize : bool
            Whether to normalize features
            
        Returns
        -------
        probabilities : np.ndarray
            Predicted probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if normalize:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            raise ValueError(f"Algorithm {self.algorithm} does not support probability prediction")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, normalize: bool = True) -> Dict[str, Any]:
        """
        Evaluate the classifier.
        
        Parameters
        ----------
        X : np.ndarray
            Test features
        y : np.ndarray
            Test labels
        normalize : bool
            Whether to normalize features
            
        Returns
        -------
        results : dict
            Evaluation results
        """
        predictions = self.predict(X, normalize=normalize)
        accuracy = accuracy_score(y, predictions)
        
        # Get unique classes for proper classification report
        unique_classes = np.unique(np.concatenate([y, predictions]))
        class_names = [str(cls) for cls in unique_classes]
        
        report = classification_report(
            y, predictions, 
            target_names=class_names, 
            output_dict=True,
            zero_division=0
        )
        
        cm = confusion_matrix(y, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5, normalize: bool = True) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        cv : int
            Number of cross-validation folds
        normalize : bool
            Whether to normalize features
            
        Returns
        -------
        cv_results : dict
            Cross-validation results
        """
        if normalize:
            # Create pipeline with scaler
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', self.model)
            ])
        else:
            pipeline = self.model
        
        # Use stratified k-fold for balanced splits
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')
        
        return {
            'cv_scores': cv_scores,
            'mean_accuracy': np.mean(cv_scores),
            'std_accuracy': np.std(cv_scores),
            'min_accuracy': np.min(cv_scores),
            'max_accuracy': np.max(cv_scores)
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: Optional[List[str]] = None,
                             title: str = 'Confusion Matrix'):
        """
        Plot confusion matrix.
        
        Parameters
        ----------
        cm : np.ndarray
            Confusion matrix
        class_names : list of str, optional
            Class names for labels
        title : str
            Plot title
        """
        plt.figure(figsize=(8, 6))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'algorithm': self.algorithm,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Parameters
        ----------
        filepath : str
            Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.algorithm = model_data['algorithm']
        self.random_state = model_data['random_state']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {filepath}")


class MultiClassifierComparison:
    """Compare multiple classifiers on the same dataset."""
    
    def __init__(self, algorithms: Optional[List[str]] = None, random_state: int = 42):
        """
        Initialize multi-classifier comparison.
        
        Parameters
        ----------
        algorithms : list of str, optional
            List of algorithms to compare
        random_state : int
            Random state for reproducibility
        """
        if algorithms is None:
            algorithms = ['svm', 'rf', 'lda', 'lr', 'knn']
        
        self.algorithms = algorithms
        self.random_state = random_state
        self.classifiers = {}
        self.results = {}
        
        # Initialize classifiers
        for alg in self.algorithms:
            self.classifiers[alg] = BCIClassifier(algorithm=alg, random_state=random_state)
    
    def compare_algorithms(self, X: np.ndarray, y: np.ndarray, 
                          cv: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Compare all algorithms using cross-validation.
        
        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        cv : int
            Number of cross-validation folds
            
        Returns
        -------
        comparison_results : dict
            Results for each algorithm
        """
        print("Comparing algorithms...")
        
        for alg_name in self.algorithms:
            print(f"Testing {alg_name.upper()}...")
            classifier = self.classifiers[alg_name]
            cv_results = classifier.cross_validate(X, y, cv=cv)
            self.results[alg_name] = cv_results
            
            print(f"  Mean accuracy: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
        
        return self.results
    
    def get_best_algorithm(self) -> Tuple[str, float]:
        """
        Get the best performing algorithm.
        
        Returns
        -------
        best_algorithm : str
            Name of best algorithm
        best_score : float
            Best mean accuracy
        """
        if not self.results:
            raise ValueError("No comparison results available. Run compare_algorithms() first.")
        
        best_alg = max(self.results.keys(), 
                      key=lambda x: self.results[x]['mean_accuracy'])
        best_score = self.results[best_alg]['mean_accuracy']
        
        return best_alg, best_score
    
    def plot_comparison(self, title: str = 'Algorithm Comparison'):
        """
        Plot comparison results.
        
        Parameters
        ----------
        title : str
            Plot title
        """
        if not self.results:
            raise ValueError("No comparison results available. Run compare_algorithms() first.")
        
        algorithms = list(self.results.keys())
        means = [self.results[alg]['mean_accuracy'] for alg in algorithms]
        stds = [self.results[alg]['std_accuracy'] for alg in algorithms]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(algorithms, means, yerr=stds, capsize=5, alpha=0.7)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{mean:.3f}±{std:.3f}',
                    ha='center', va='bottom')
        
        plt.title(title)
        plt.xlabel('Algorithm')
        plt.ylabel('Cross-Validation Accuracy')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()