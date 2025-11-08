"""
Feature extraction utilities for BCI classification
"""

from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
from mne.decoding import CSP
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from preprocessed EEG epochs for BCI classification."""
    
    def __init__(self, method: str = 'csp', n_components: int = 4):
        """
        Initialize FeatureExtractor.
        
        Parameters
        ----------
        method : str
            Feature extraction method ('csp', 'psd', 'time_domain', 'combined')
        n_components : int
            Number of components for CSP
        """
        self.method = method
        self.n_components = n_components
        self.csp = None
        self.feature_names_ = []
        
    def extract_csp_features(self, epochs: mne.Epochs, labels: np.ndarray) -> np.ndarray:
        """
        Extract Common Spatial Pattern (CSP) features.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Preprocessed epochs
        labels : np.ndarray
            Class labels
            
        Returns
        -------
        features : np.ndarray
            CSP features (n_trials, n_components)
        """
        # Get data
        X = epochs.get_data()  # (n_trials, n_channels, n_times)
        
        # Initialize and fit CSP
        self.csp = CSP(n_components=self.n_components, reg=None, log=True, norm_trace=False)
        features = self.csp.fit_transform(X, labels)
        
        # Update feature names
        self.feature_names_ = [f'CSP_{i}' for i in range(self.n_components)]
        
        return features
    
    def extract_psd_features(self, epochs: mne.Epochs, 
                           freq_bands: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
        """
        Extract Power Spectral Density (PSD) features.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Preprocessed epochs
        freq_bands : dict, optional
            Frequency bands for feature extraction
            
        Returns
        -------
        features : np.ndarray
            PSD features
        """
        if freq_bands is None:
            freq_bands = {
                'alpha': (8, 12),
                'beta': (13, 30),
                'gamma': (31, 45),
                'theta': (4, 7)
            }
        
        # Get data
        X = epochs.get_data()  # (n_trials, n_channels, n_times)
        sfreq = epochs.info['sfreq']
        
        features_list = []
        feature_names = []
        
        for trial in X:
            trial_features = []
            for ch_idx, ch_name in enumerate(epochs.ch_names):
                # Compute PSD for each frequency band
                for band_name, (fmin, fmax) in freq_bands.items():
                    freqs, psd = signal.welch(trial[ch_idx], sfreq, nperseg=int(sfreq))
                    freq_mask = (freqs >= fmin) & (freqs <= fmax)
                    band_power = np.mean(psd[freq_mask])
                    trial_features.append(band_power)
                    
                    if len(features_list) == 0:  # First trial, create feature names
                        feature_names.append(f'{ch_name}_{band_name}_power')
            
            features_list.append(trial_features)
        
        features = np.array(features_list)
        self.feature_names_ = feature_names
        
        return features
    
    def extract_time_domain_features(self, epochs: mne.Epochs) -> np.ndarray:
        """
        Extract time-domain statistical features.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Preprocessed epochs
            
        Returns
        -------
        features : np.ndarray
            Time-domain features
        """
        # Get data
        X = epochs.get_data()  # (n_trials, n_channels, n_times)
        
        features_list = []
        feature_names = []
        
        for trial in X:
            trial_features = []
            for ch_idx, ch_name in enumerate(epochs.ch_names):
                ch_data = trial[ch_idx]
                
                # Statistical features
                features_dict = {
                    'mean': np.mean(ch_data),
                    'std': np.std(ch_data),
                    'var': np.var(ch_data),
                    'skew': self._skewness(ch_data),
                    'kurt': self._kurtosis(ch_data),
                    'rms': np.sqrt(np.mean(ch_data**2)),
                    'peak_to_peak': np.ptp(ch_data),
                }
                
                for feat_name, feat_val in features_dict.items():
                    trial_features.append(feat_val)
                    if len(features_list) == 0:  # First trial, create feature names
                        feature_names.append(f'{ch_name}_{feat_name}')
            
            features_list.append(trial_features)
        
        features = np.array(features_list)
        self.feature_names_ = feature_names
        
        return features
    
    def extract_combined_features(self, epochs: mne.Epochs, labels: np.ndarray) -> np.ndarray:
        """
        Extract combined features (CSP + PSD + Time-domain).
        
        Parameters
        ----------
        epochs : mne.Epochs
            Preprocessed epochs
        labels : np.ndarray
            Class labels
            
        Returns
        -------
        features : np.ndarray
            Combined features
        """
        # Extract different types of features
        csp_features = self.extract_csp_features(epochs, labels)
        psd_features = self.extract_psd_features(epochs)
        time_features = self.extract_time_domain_features(epochs)
        
        # Combine features
        features = np.hstack([csp_features, psd_features, time_features])
        
        # Update feature names
        csp_names = [f'CSP_{i}' for i in range(self.n_components)]
        self.feature_names_ = csp_names + self.feature_names_ + [f'time_{name}' for name in self.feature_names_[-time_features.shape[1]:]]
        
        return features
    
    def fit(self, epochs: mne.Epochs, labels: np.ndarray):
        """
        Fit the feature extractor.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Training epochs
        labels : np.ndarray
            Training labels
            
        Returns
        -------
        self : FeatureExtractor
            Fitted feature extractor
        """
        if self.method == 'csp':
            self.extract_csp_features(epochs, labels)
        elif self.method == 'combined':
            self.extract_combined_features(epochs, labels)
        # For PSD and time_domain, no fitting is required
        
        return self
    
    def transform(self, epochs: mne.Epochs, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform epochs to features.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Input epochs
        labels : np.ndarray, optional
            Labels (required for CSP)
            
        Returns
        -------
        features : np.ndarray
            Extracted features
        """
        if self.method == 'csp':
            if self.csp is None:
                raise ValueError("CSP not fitted. Call fit() first.")
            X = epochs.get_data()
            return self.csp.transform(X)
        
        elif self.method == 'psd':
            return self.extract_psd_features(epochs)
        
        elif self.method == 'time_domain':
            return self.extract_time_domain_features(epochs)
        
        elif self.method == 'combined':
            if labels is None:
                raise ValueError("Labels required for combined features with CSP.")
            return self.extract_combined_features(epochs, labels)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def fit_transform(self, epochs: mne.Epochs, labels: np.ndarray) -> np.ndarray:
        """
        Fit and transform epochs to features.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Training epochs
        labels : np.ndarray
            Training labels
            
        Returns
        -------
        features : np.ndarray
            Extracted features
        """
        return self.fit(epochs, labels).transform(epochs, labels)
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names.
        
        Returns
        -------
        feature_names : list of str
            Feature names
        """
        return self.feature_names_
    
    def _skewness(self, x: np.ndarray) -> float:
        """Calculate skewness of signal."""
        mean_x = np.mean(x)
        std_x = np.std(x)
        return np.mean(((x - mean_x) / std_x) ** 3) if std_x != 0 else 0.0
    
    def _kurtosis(self, x: np.ndarray) -> float:
        """Calculate kurtosis of signal."""
        mean_x = np.mean(x)
        std_x = np.std(x)
        return np.mean(((x - mean_x) / std_x) ** 4) - 3 if std_x != 0 else 0.0


class ERPFeatureExtractor:
    """Extract Event-Related Potential (ERP) features."""
    
    def __init__(self, peak_windows: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize ERP feature extractor.
        
        Parameters
        ----------
        peak_windows : dict, optional
            Time windows for ERP components (e.g., P300, N200)
        """
        if peak_windows is None:
            peak_windows = {
                'P300': (0.25, 0.45),  # P300 component
                'N200': (0.15, 0.25),  # N200 component
                'P200': (0.15, 0.25),  # P200 component
            }
        self.peak_windows = peak_windows
    
    def extract_erp_features(self, epochs: mne.Epochs, 
                           channels: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract ERP features from epochs.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Epoched ERP data
        channels : list of str, optional
            Channels to extract features from
            
        Returns
        -------
        features : np.ndarray
            ERP features
        """
        if channels is None:
            channels = ['Cz', 'Pz', 'Oz']  # Common ERP channels
        
        # Pick available channels
        available_channels = [ch for ch in channels if ch in epochs.ch_names]
        if not available_channels:
            available_channels = epochs.ch_names[:3]  # Use first 3 channels
        
        epochs_picked = epochs.copy().pick_channels(available_channels)
        X = epochs_picked.get_data()  # (n_trials, n_channels, n_times)
        times = epochs_picked.times
        
        features_list = []
        
        for trial in X:
            trial_features = []
            
            for ch_idx, ch_name in enumerate(available_channels):
                ch_data = trial[ch_idx]
                
                # Extract features for each ERP component
                for component, (tmin, tmax) in self.peak_windows.items():
                    time_mask = (times >= tmin) & (times <= tmax)
                    
                    if np.any(time_mask):
                        window_data = ch_data[time_mask]
                        
                        # Peak amplitude and latency
                        peak_idx = np.argmax(np.abs(window_data))
                        peak_amplitude = window_data[peak_idx]
                        peak_latency = times[time_mask][peak_idx]
                        
                        # Mean amplitude in window
                        mean_amplitude = np.mean(window_data)
                        
                        trial_features.extend([peak_amplitude, peak_latency, mean_amplitude])
                    else:
                        # If time window not available, add zeros
                        trial_features.extend([0.0, 0.0, 0.0])
            
            features_list.append(trial_features)
        
        return np.array(features_list)