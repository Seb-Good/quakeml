"""
training_dataset.py
-------------------
This module provides classes and methods for creating a training dataset.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import json
import shutil
import zipfile
import numpy as np
import pandas as pd
from scipy.signal import argrelmax
from joblib import Parallel, delayed

# Local imports
from quakeml.config.config import DATA_DIR


class TrainingDataset(object):

    """
    LANL Earthquake Prediction
    Can you predict upcoming laboratory earthquakes?
    https://www.kaggle.com/c/LANL-Earthquake-Prediction/data
    """

    def __init__(self, train_intervals, val_intervals, overlap):

        # Set parameters
        self.train_intervals = train_intervals
        self.val_intervals = val_intervals
        self.overlap = overlap

        # Set attributes
        self.fs = 330000
        self.length = 150000
        self.processed_path = os.path.join(DATA_DIR, 'processed')
        self.training_path = os.path.join(DATA_DIR, 'training')
        self.meta_data = json.load(open(os.path.join(self.processed_path, 'meta_data.json')))

    def generate_db(self):
        """Create training and validation datasets."""
        # Training dataset
        self._create_train_db()

        # Validation dataset
        self._create_val_db()

    def _create_train_db(self):
        """Create training dataset."""
        pass

    def _create_val_db(self):
        """Create training dataset."""
        pass

    def _save_waveform(self, waveform, ):
        """Save waveform as npy file."""
