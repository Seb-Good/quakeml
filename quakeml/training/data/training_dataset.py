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

        # Get intervals
        self.intervals = self._get_intervals()

        # Add time to failure
        # self.intervals = self._add_time_to_failure()

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

    def _get_intervals(self):
        """Collect intervals in DataFrame."""
        # List for intervals
        intervals = list()

        # Loop through cycles
        for cycle_id in [int(cycle) for cycle in self.meta_data['cycles'].keys()]:

            # Import time_to_failure array
            waveform = np.load(os.path.join(self.processed_path, 'acoustic_data_cycle_{}.npy').format(cycle_id))

            # Get intervals
            intervals.extend(self._calculate_interval_bounds(waveform=waveform, cycle_id=cycle_id))

        # Convert to DataFrame
        df = pd.DataFrame(intervals, columns=['cycle_id', 'start', 'end'])\
            .sort_values(by=['cycle_id', 'start']).reset_index(drop=True)

        # Add file names
        df['time_to_failure_file'] = \
            df.apply(lambda row: self.meta_data['cycles'][str(row['cycle_id'])]['time_to_failure'], axis=1)
        df['acoustic_data_file'] = \
            df.apply(lambda row: self.meta_data['cycles'][str(row['cycle_id'])]['acoustic_data'], axis=1)

        return df

    def _calculate_interval_bounds(self, cycle_id, waveform):
        """Calculate the start and end sample point for each interval in a cycle."""
        # Set index
        idx = 0

        # List for intervals
        intervals = list()

        while True:
            if idx + self.length <= len(waveform) - 1:
                intervals.append((cycle_id, idx, idx + self.length))
                idx += int(self.length - self.length * self.overlap)
            elif idx + self.length > len(waveform) - 1:
                intervals.append((cycle_id, len(waveform) - 1 - self.length, len(waveform) - 1))
                break

        return intervals

    def _add_time_to_failure(self):
        """Extract the time to failure for each interval."""
        pass

    def _save_waveform(self, file_name, waveform, dataset):
        """Save waveform as npy file."""
        np.save(os.path.join(self.training_path, dataset, 'waveforms', file_name), waveform)
