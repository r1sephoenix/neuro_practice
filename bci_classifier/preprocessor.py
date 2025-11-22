"""
Preprocessing utilities for BCI data using MNE Python
"""

from typing import List, Optional, Tuple, Union

import mne
import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """Preprocess BCI data including filtering, artifact removal, and normalization."""
    
    def __init__(self):
        """Initialize Preprocessor."""
        self.scaler = StandardScaler()
        self.ica = None
        
    def filter_data(self, epochs: mne.Epochs, l_freq: float = 8.0, h_freq: float = 30.0) -> mne.Epochs:
        """
        Apply band-pass filter to epochs.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Input epochs
        l_freq : float
            Low frequency for high-pass filter (Hz)
        h_freq : float
            High frequency for low-pass filter (Hz)
            
        Returns
        -------
        epochs_filtered : mne.Epochs
            Filtered epochs
        """
        epochs_filtered = epochs.copy()
        epochs_filtered.filter(l_freq, h_freq, fir_design='firwin', verbose=False)
        return epochs_filtered
    
    def apply_ica(self, epochs: mne.Epochs, n_components: int = 15, 
                  random_state: int = 42) -> Tuple[mne.Epochs, mne.preprocessing.ICA]:
        """
        Apply Independent Component Analysis (ICA) for artifact removal.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Input epochs
        n_components : int
            Number of ICA components
        random_state : int
            Random state for reproducibility
            
        Returns
        -------
        epochs_clean : mne.Epochs
            Epochs after ICA artifact removal
        ica : mne.preprocessing.ICA
            Fitted ICA object
        """
        # Fit ICA
        self.ica = mne.preprocessing.ICA(
            n_components=n_components,
            random_state=random_state,
            verbose=False
        )
        
        # Copy epochs for processing
        epochs_copy = epochs.copy()
        self.ica.fit(epochs_copy)
        
        # # Find and exclude EOG artifacts automatically
        # eog_indices, eog_scores = self.ica.find_bads_eog(epochs_copy, verbose=False)
        # self.ica.exclude = eog_indices
        
        # Apply ICA to remove artifacts
        epochs_clean = self.ica.apply(epochs.copy(), verbose=False)
        
        return epochs_clean, self.ica
    
    def reject_bad_epochs(self, epochs: mne.Epochs, 
                         reject_criteria: Optional[dict] = None) -> mne.Epochs:
        """
        Reject bad epochs based on amplitude criteria.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Input epochs
        reject_criteria : dict, optional
            Rejection criteria. Default uses standard values for EEG.
            
        Returns
        -------
        epochs_clean : mne.Epochs after rejection
        """
        if reject_criteria is None:
            reject_criteria = {
                'eeg': 150e-6,  # 150 µV
                # 'eog': 250e-6,  # 250 µV
            }
        
        epochs_clean = epochs.copy()
        epochs_clean.drop_bad(reject=reject_criteria, verbose=False)
        
        return epochs_clean
    
    def apply_car(self, epochs: mne.Epochs) -> mne.Epochs:
        """
        Apply Common Average Reference (CAR) to epochs.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Input epochs
            
        Returns
        -------
        epochs_car : mne.Epochs
            Epochs with CAR applied
        """
        epochs_car = epochs.copy()
        epochs_car.set_eeg_reference('average', projection=True, verbose=False)
        epochs_car.apply_proj(verbose=False)
        
        return epochs_car
    
    def downsample_epochs(self, epochs: mne.Epochs, target_sfreq: float) -> mne.Epochs:
        """
        Downsample epochs to target sampling frequency.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Input epochs
        target_sfreq : float
            Target sampling frequency
            
        Returns
        -------
        epochs_downsampled : mne.Epochs
            Downsampled epochs
        """
        epochs_downsampled = epochs.copy()
        epochs_downsampled.resample(target_sfreq, verbose=False)
        
        return epochs_downsampled
    
    def normalize_data(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize data using StandardScaler.
        
        Parameters
        ----------
        X : np.ndarray
            Input data (n_samples, n_features)
        fit : bool
            Whether to fit the scaler or use previously fitted scaler
            
        Returns
        -------
        X_normalized : np.ndarray
            Normalized data
        """
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def preprocess_epochs(self, epochs: mne.Epochs,
                         l_freq: float = 8.0,
                         h_freq: float = 30.0,
                         apply_ica: bool = True,
                         apply_car: bool = True,
                         target_sfreq: Optional[float] = None,
                         reject_bad: bool = True) -> mne.Epochs:
        """
        Apply full preprocessing pipeline to epochs.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Input epochs
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
            
        Returns
        -------
        epochs_processed : mne.Epochs
            Preprocessed epochs
        """
        print("Starting preprocessing pipeline...")
        
        # Filter data
        print(f"Applying band-pass filter ({l_freq}-{h_freq} Hz)...")
        epochs_processed = self.filter_data(epochs, l_freq, h_freq)
        
        # Apply CAR if requested
        if apply_car:
            print("Applying Common Average Reference...")
            epochs_processed = self.apply_car(epochs_processed)
        
        # Reject bad epochs if requested
        if reject_bad:
            print("Rejecting bad epochs...")
            n_epochs_before = len(epochs_processed)
            epochs_processed = self.reject_bad_epochs(epochs_processed)
            n_epochs_after = len(epochs_processed)
            print(f"Rejected {n_epochs_before - n_epochs_after} bad epochs")
        
        # Apply ICA if requested
        if apply_ica:
            print("Applying ICA for artifact removal...")
            epochs_processed, _ = self.apply_ica(epochs_processed)
        
        # Downsample if requested
        if target_sfreq is not None:
            print(f"Downsampling to {target_sfreq} Hz...")
            epochs_processed = self.downsample_epochs(epochs_processed, target_sfreq)
        
        print("Preprocessing completed!")
        return epochs_processed
    
    def get_preprocessing_info(self) -> dict:
        """
        Get information about applied preprocessing steps.
        
        Returns
        -------
        info : dict
            Preprocessing information
        """
        info = {
            "scaler_fitted": hasattr(self.scaler, 'mean_'),
            "ica_fitted": self.ica is not None,
        }
        
        if self.ica is not None:
            info.update({
                "ica_components": self.ica.n_components_,
                "excluded_components": len(self.ica.exclude),
            })
        
        return info