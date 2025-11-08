"""
BCI Classifier - Brain-Computer Interface classification using MNE Python
"""

__version__ = "0.1.0"
__author__ = "BCI Classifier Team"

from .classifier import BCIClassifier, MultiClassifierComparison
from .data_loader import DataLoader
from .feature_extractor import FeatureExtractor
from .pipeline import BCIPipeline
from .preprocessor import Preprocessor

__all__ = [
    "DataLoader",
    "Preprocessor", 
    "FeatureExtractor",
    "BCIClassifier",
    "MultiClassifierComparison",
    "BCIPipeline",
]