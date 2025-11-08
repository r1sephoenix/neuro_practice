"""
Complete BCI classification pipeline
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np

from .classifier import BCIClassifier, MultiClassifierComparison
from .data_loader import DataLoader
from .feature_extractor import FeatureExtractor
from .preprocessor import Preprocessor


class BCIPipeline:
    """Complete BCI classification pipeline from data loading to classification."""
    
    def __init__(self, 
                 dataset: str = 'eegbci',
                 feature_method: str = 'csp',
                 classifier_algorithm: str = 'svm',
                 random_state: int = 42):
        """
        Initialize BCI pipeline.
        
        Parameters
        ----------
        dataset : str
            Dataset to use ('eegbci', 'sample')
        feature_method : str
            Feature extraction method ('csp', 'psd', 'time_domain', 'combined')
        classifier_algorithm : str
            Classification algorithm ('svm', 'rf', 'lda', 'lr', 'knn')
        random_state : int
            Random state for reproducibility
        """
        self.dataset = dataset
        self.feature_method = feature_method
        self.classifier_algorithm = classifier_algorithm
        self.random_state = random_state
        
        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.feature_extractor = FeatureExtractor(method=feature_method)
        self.classifier = BCIClassifier(algorithm=classifier_algorithm, 
                                       random_state=random_state)
        
        # Data storage
        self.raw_data = None
        self.events = None
        self.event_id = None
        self.epochs = None
        self.epochs_train = None
        self.epochs_test = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Results storage
        self.results = {}
    
    def load_data(self, subject: int = 1, **kwargs) -> None:
        """
        Load dataset.
        
        Parameters
        ----------
        subject : int
            Subject number (for datasets that support multiple subjects)
        **kwargs
            Additional arguments for data loading
        """
        print(f"Loading {self.dataset} dataset...")
        
        if self.dataset == 'eegbci':
            self.raw_data, self.events, self.event_id = self.data_loader.load_eegbci_data(
                subject=subject, **kwargs
            )
        elif self.dataset == 'sample':
            self.raw_data, self.events, self.event_id = self.data_loader.load_sample_data()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        
        print(f"Data loaded successfully!")
        print(f"Dataset info: {self.data_loader.get_info()}")
    
    def create_epochs(self, tmin: float = -1.0, tmax: float = 4.0,
                     baseline: Optional[Tuple[float, float]] = (-1.0, 0.0),
                     picks: Optional[List[str]] = None) -> None:
        """
        Create epochs from raw data.
        
        Parameters
        ----------
        tmin : float
            Start time before event (in s)
        tmax : float
            End time after event (in s)
        baseline : tuple or None
            Baseline correction period
        picks : list of str or None
            Channels to pick
        """
        print("Creating epochs...")
        
        self.epochs = self.data_loader.get_epochs(
            tmin=tmin, tmax=tmax, baseline=baseline, picks=picks
        )
        
        print(f"Created {len(self.epochs)} epochs")
        print(f"Epoch shape: {self.epochs.get_data().shape}")
    
    def preprocess_data(self, 
                       l_freq: float = 8.0,
                       h_freq: float = 30.0,
                       apply_ica: bool = True,
                       apply_car: bool = True,
                       target_sfreq: Optional[float] = None,
                       reject_bad: bool = True) -> None:
        """
        Preprocess epochs.
        
        Parameters
        ----------
        l_freq : float
            Low frequency for band-pass filter
        h_freq : float
            High frequency for band-pass filter
        apply_ica : bool
            Whether to apply ICA for artifact removal
        apply_car : bool
            Whether to apply Common Average Reference
        target_sfreq : float, optional
            Target sampling frequency for downsampling
        reject_bad : bool
            Whether to reject bad epochs
        """
        if self.epochs is None:
            raise ValueError("No epochs available. Call create_epochs() first.")
        
        print("Preprocessing data...")
        
        self.epochs = self.preprocessor.preprocess_epochs(
            self.epochs,
            l_freq=l_freq,
            h_freq=h_freq,
            apply_ica=apply_ica,
            apply_car=apply_car,
            target_sfreq=target_sfreq,
            reject_bad=reject_bad
        )
        
        print(f"Preprocessing completed. Final epochs: {len(self.epochs)}")
    
    def split_data(self, test_size: float = 0.2, stratify: bool = True) -> None:
        """
        Split data into training and testing sets.
        
        Parameters
        ----------
        test_size : float
            Proportion of data for testing
        stratify : bool
            Whether to stratify the split
        """
        if self.epochs is None:
            raise ValueError("No epochs available. Preprocess data first.")
        
        from sklearn.model_selection import train_test_split

        # Get labels from epochs
        labels = self.epochs.events[:, -1]
        
        # Split epochs indices
        n_epochs = len(self.epochs)
        indices = np.arange(n_epochs)
        
        if stratify:
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, stratify=labels, 
                random_state=self.random_state
            )
        else:
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=self.random_state
            )
        
        # Split epochs
        self.epochs_train = self.epochs[train_idx]
        self.epochs_test = self.epochs[test_idx]
        
        # Get labels
        self.y_train = self.epochs_train.events[:, -1]
        self.y_test = self.epochs_test.events[:, -1]
        
        print(f"Data split: {len(self.epochs_train)} training, {len(self.epochs_test)} testing epochs")
        print(f"Class distribution - Train: {np.bincount(self.y_train)}, Test: {np.bincount(self.y_test)}")
    
    def extract_features(self) -> None:
        """Extract features from training and testing epochs."""
        if self.epochs_train is None or self.epochs_test is None:
            raise ValueError("No split data available. Call split_data() first.")
        
        print(f"Extracting features using {self.feature_method} method...")
        
        # Fit and transform training data
        self.X_train = self.feature_extractor.fit_transform(self.epochs_train, self.y_train)
        
        # Transform testing data
        if self.feature_method in ['csp', 'combined']:
            self.X_test = self.feature_extractor.transform(self.epochs_test, self.y_test)
        else:
            self.X_test = self.feature_extractor.transform(self.epochs_test)
        
        print(f"Features extracted - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        
        # Print feature names if available
        if hasattr(self.feature_extractor, 'get_feature_names'):
            feature_names = self.feature_extractor.get_feature_names()
            if feature_names:
                print(f"Feature names: {feature_names[:5]}..." if len(feature_names) > 5 else f"Feature names: {feature_names}")
    
    def train_classifier(self, normalize: bool = True) -> None:
        """
        Train the classifier.
        
        Parameters
        ----------
        normalize : bool
            Whether to normalize features
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training features available. Call extract_features() first.")
        
        print(f"Training {self.classifier_algorithm} classifier...")
        
        self.classifier.fit(self.X_train, self.y_train, normalize=normalize)
        
        print("Classifier trained successfully!")
    
    def evaluate_classifier(self, normalize: bool = True) -> Dict[str, Any]:
        """
        Evaluate the classifier.
        
        Parameters
        ----------
        normalize : bool
            Whether to normalize features
            
        Returns
        -------
        results : dict
            Evaluation results
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError("No test features available. Call extract_features() first.")
        
        print("Evaluating classifier...")
        
        # Evaluate on test set
        test_results = self.classifier.evaluate(self.X_test, self.y_test, normalize=normalize)
        
        # Cross-validation on training set
        cv_results = self.classifier.cross_validate(self.X_train, self.y_train, normalize=normalize)
        
        self.results = {
            'test_accuracy': test_results['accuracy'],
            'test_classification_report': test_results['classification_report'],
            'test_confusion_matrix': test_results['confusion_matrix'],
            'cv_mean_accuracy': cv_results['mean_accuracy'],
            'cv_std_accuracy': cv_results['std_accuracy'],
            'cv_scores': cv_results['cv_scores']
        }
        
        print(f"Test Accuracy: {test_results['accuracy']:.3f}")
        print(f"CV Accuracy: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
        
        return self.results
    
    def run_complete_pipeline(self,
                             subject: int = 1,
                             tmin: float = -1.0,
                             tmax: float = 4.0,
                             l_freq: float = 8.0,
                             h_freq: float = 30.0,
                             test_size: float = 0.2,
                             apply_ica: bool = True,
                             normalize_features: bool = True,
                             **kwargs) -> Dict[str, Any]:
        """
        Run the complete BCI classification pipeline.
        
        Parameters
        ----------
        subject : int
            Subject number
        tmin, tmax : float
            Epoch time limits
        l_freq, h_freq : float
            Filter frequencies
        test_size : float
            Test set proportion
        apply_ica : bool
            Whether to apply ICA
        normalize_features : bool
            Whether to normalize features
        **kwargs
            Additional parameters
            
        Returns
        -------
        results : dict
            Pipeline results
        """
        print("="*50)
        print("RUNNING COMPLETE BCI CLASSIFICATION PIPELINE")
        print("="*50)
        
        try:
            # Step 1: Load data
            self.load_data(subject=subject, **kwargs)
            
            # Step 2: Create epochs
            self.create_epochs(tmin=tmin, tmax=tmax)
            
            # Step 3: Preprocess data
            self.preprocess_data(l_freq=l_freq, h_freq=h_freq, apply_ica=apply_ica)
            
            # Step 4: Split data
            self.split_data(test_size=test_size)
            
            # Step 5: Extract features
            self.extract_features()
            
            # Step 6: Train classifier
            self.train_classifier(normalize=normalize_features)
            
            # Step 7: Evaluate classifier
            results = self.evaluate_classifier(normalize=normalize_features)
            
            print("="*50)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*50)
            
            return results
            
        except Exception as e:
            print(f"Pipeline failed with error: {str(e)}")
            raise
    
    def compare_algorithms(self, algorithms: Optional[List[str]] = None,
                          cv: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Compare different classification algorithms.
        
        Parameters
        ----------
        algorithms : list of str, optional
            Algorithms to compare
        cv : int
            Number of cross-validation folds
            
        Returns
        -------
        comparison_results : dict
            Results for each algorithm
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training features available. Run pipeline first.")
        
        print("Comparing classification algorithms...")
        
        comparison = MultiClassifierComparison(algorithms=algorithms, 
                                             random_state=self.random_state)
        results = comparison.compare_algorithms(self.X_train, self.y_train, cv=cv)
        
        # Get best algorithm
        best_alg, best_score = comparison.get_best_algorithm()
        print(f"\nBest algorithm: {best_alg.upper()} (accuracy: {best_score:.3f})")
        
        # Plot comparison
        comparison.plot_comparison()
        
        return results
    
    def plot_results(self):
        """Plot classification results."""
        if not self.results:
            raise ValueError("No results available. Run evaluation first.")
        
        # Plot confusion matrix
        cm = self.results['test_confusion_matrix']
        
        # Get class names from event_id
        class_names = list(self.event_id.keys()) if self.event_id else None
        
        self.classifier.plot_confusion_matrix(
            cm, class_names=class_names, 
            title=f'{self.classifier_algorithm.upper()} Confusion Matrix'
        )
    
    def save_pipeline(self, filepath: str):
        """
        Save the complete pipeline.
        
        Parameters
        ----------
        filepath : str
            Path to save the pipeline
        """
        pipeline_data = {
            'dataset': self.dataset,
            'feature_method': self.feature_method,
            'classifier_algorithm': self.classifier_algorithm,
            'random_state': self.random_state,
            'feature_extractor': self.feature_extractor,
            'classifier': self.classifier,
            'preprocessor': self.preprocessor,
            'event_id': self.event_id,
            'results': self.results
        }
        
        joblib.dump(pipeline_data, filepath)
        print(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """
        Load a saved pipeline.
        
        Parameters
        ----------
        filepath : str
            Path to load the pipeline from
        """
        pipeline_data = joblib.load(filepath)
        
        self.dataset = pipeline_data['dataset']
        self.feature_method = pipeline_data['feature_method']
        self.classifier_algorithm = pipeline_data['classifier_algorithm']
        self.random_state = pipeline_data['random_state']
        self.feature_extractor = pipeline_data['feature_extractor']
        self.classifier = pipeline_data['classifier']
        self.preprocessor = pipeline_data['preprocessor']
        self.event_id = pipeline_data['event_id']
        self.results = pipeline_data.get('results', {})
        
        print(f"Pipeline loaded from {filepath}")
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get summary of pipeline configuration and results.
        
        Returns
        -------
        summary : dict
            Pipeline summary
        """
        summary = {
            'configuration': {
                'dataset': self.dataset,
                'feature_method': self.feature_method,
                'classifier_algorithm': self.classifier_algorithm,
                'random_state': self.random_state,
            },
            'data_info': self.data_loader.get_info() if self.raw_data is not None else None,
            'preprocessing_info': self.preprocessor.get_preprocessing_info(),
            'results': self.results
        }
        
        if self.X_train is not None:
            summary['feature_info'] = {
                'n_features': self.X_train.shape[1],
                'n_train_samples': self.X_train.shape[0],
                'n_test_samples': self.X_test.shape[0] if self.X_test is not None else None
            }
        
        return summary