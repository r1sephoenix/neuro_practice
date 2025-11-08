"""
Data loader for BCI datasets using MNE Python
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import mne
import numpy as np


class DataLoader:
    """Load and handle BCI datasets using MNE Python."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize DataLoader.
        
        Parameters
        ----------
        data_path : str, optional
            Path to local data directory
        """
        self.data_path = Path(data_path) if data_path else None
        self.raw = None
        self.events = None
        self.event_id = None
        
    def load_sample_data(self) -> Tuple[mne.io.Raw, np.ndarray, dict]:
        """
        Load MNE sample dataset for demonstration.
        
        Returns
        -------
        raw : mne.io.Raw
            Raw EEG data
        events : np.ndarray
            Events array
        event_id : dict
            Event ID dictionary
        """
        # Download sample data if not available
        data_path = mne.datasets.sample.data_path()
        raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'
        event_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw-eve.fif'
        
        # Load raw data
        self.raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
        
        # Load events
        self.events = mne.read_events(event_fname, verbose=False)
        
        # Define event IDs for auditory/visual stimuli
        self.event_id = {
            'auditory/left': 1,
            'auditory/right': 2,
            'visual/left': 3,
            'visual/right': 4
        }
        
        return self.raw, self.events, self.event_id
    
    def load_eegbci_data(self, subject: int = 1, runs: List[int] = [6, 10, 14]) -> Tuple[mne.io.Raw, np.ndarray, dict]:
        """
        Load EEG Motor Movement/Imagery Dataset from PhysioNet.
        
        Parameters
        ----------
        subject : int
            Subject number (1-109)
        runs : list of int
            List of runs to load. Default: [6, 10, 14] (left vs right hand motor imagery)
            
        Returns
        -------
        raw : mne.io.Raw
            Raw EEG data
        events : np.ndarray
            Events array
        event_id : dict
            Event ID dictionary
        """
        from mne.datasets import eegbci
        from mne.io import concatenate_raws

        # Download data
        eegbci.load_data(subject, runs, update_path=True)
        
        # Load raw files
        raw_fnames = eegbci.load_data(subject, runs)
        raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
        
        # Concatenate runs
        self.raw = concatenate_raws(raws)
        
        # Rename channels to standard 10-05 system
        mne.datasets.eegbci.standardize(self.raw)
        
        # Apply montage
        montage = mne.channels.make_standard_montage('standard_1005')
        self.raw.set_montage(montage, verbose=False)
        
        # Find events from annotations (EEGBCI data stores events as annotations)
        try:
            self.events, event_id_from_annotations = mne.events_from_annotations(self.raw, verbose=False)
        except ValueError as e:
            if "Missing channels from ch_names required by include" in str(e) and "STI 014" in str(e):
                # Handle missing stimulus channels by creating events without stimulus channel requirement
                # Use event_id parameter to specify which annotations to convert to events
                event_id_mapping = {'T1': 2, 'T2': 3, 'T0': 1}  # Common EEGBCI event mappings
                self.events, event_id_from_annotations = mne.events_from_annotations(
                    self.raw, event_id=event_id_mapping, verbose=False
                )
            else:
                raise e
        
        # Event IDs for motor imagery
        self.event_id = {
            'left_hand': 2,
            'right_hand': 3,
            'rest': 1
        }
        
        return self.raw, self.events, self.event_id
    
    def get_epochs(self, tmin: float = -1.0, tmax: float = 4.0, 
                   baseline: Optional[Tuple[float, float]] = (-1.0, 0.0),
                   picks: Optional[List[str]] = None) -> mne.Epochs:
        """
        Create epochs from raw data and events.
        
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
            
        Returns
        -------
        epochs : mne.Epochs
            Epoched data
        """
        if self.raw is None or self.events is None:
            raise ValueError("No data loaded. Use load_sample_data() or load_eegbci_data() first.")
        
        # Pick EEG channels if not specified
        if picks is None:
            picks = 'eeg'
        
        epochs = mne.Epochs(
            self.raw, 
            self.events, 
            self.event_id,
            tmin=tmin, 
            tmax=tmax,
            baseline=baseline,
            picks=picks,
            preload=True,
            verbose=False
        )
        
        return epochs
    
    def get_info(self) -> dict:
        """
        Get information about loaded data.
        
        Returns
        -------
        info : dict
            Data information
        """
        if self.raw is None:
            return {"status": "No data loaded"}
        
        return {
            "n_channels": len(self.raw.ch_names),
            "channel_names": self.raw.ch_names,
            "sampling_rate": self.raw.info['sfreq'],
            "duration": self.raw.times[-1],
            "n_events": len(self.events) if self.events is not None else 0,
            "event_types": list(self.event_id.keys()) if self.event_id else []
        }