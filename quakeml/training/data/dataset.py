"""
dataset.py
----------
This module provides classes and methods for compiling the LANL Earthquake Prediction dataset.
By: Sebastian D. Goodfellow, Ph.D., 2018
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import os
import shutil
import zipfile
import numpy as np
import pandas as pd
from scipy.signal import argrelmax
from joblib import Parallel, delayed

# Local imports
from quakeml.config.config import DATA_DIR


class Dataset(object):

    """
    LANL Earthquake Prediction
    Can you predict upcoming laboratory earthquakes?
    https://www.kaggle.com/c/LANL-Earthquake-Prediction/data
    """

    def __init__(self):

        # Set attributes
        self.raw_path = os.path.join(DATA_DIR, 'raw')
        self.processed_path = os.path.join(DATA_DIR, 'processed')
        self.training_path = os.path.join(DATA_DIR, 'training')

    def generate_db(self):
        """Generate the LANL Earthquake Prediction database in the 'processed' folder."""
        print('Generating LANL Earthquake Prediction Database...')
        # Unzip the database
        self._unzip_db()

        # Get test dataset
        self._get_test_db()

        # Get train dataset
        self._get_train_db()
        print('Complete!\n')

    def _unzip_db(self):
        """Unzip the raw db zip file."""
        print('Unzipping database...')
        with zipfile.ZipFile(os.path.join(self.raw_path, 'all.zip'), 'r') as zip_ref:
            zip_ref.extractall(self.processed_path)

    def _get_test_db(self):
        """Create the test dataset and convert from csv to npy files."""
        print('Creating test database...')
        # Unzip test dataset
        self._unzip_test_db()

        # Get list of files
        file_names = os.listdir(os.path.join(self.processed_path, 'test'))

        # Move test files to training folder
        _ = Parallel(n_jobs=-1)(delayed(self._csv_to_npy)(file_name) for file_name in file_names[0:10])

        # Remove test directory
        shutil.rmtree(os.path.join(self.processed_path, 'test'))

        # Remove zip file
        os.remove(os.path.join(self.processed_path, 'test.zip'))

    def _get_train_db(self):
        """Format training data."""
        print('Creating train database...')
        # import training data
        acoustic_data, time_to_failure = self._import_train_data()

        # Get loading cycle intervals
        cycle_intervals = self._get_loading_cycles(time_to_failure=time_to_failure)

        # Save cycles
        self._save_loading_cycles(acoustic_data=acoustic_data, time_to_failure=time_to_failure,
                                  cycle_intervals=cycle_intervals)

        # Delete train.csv
        os.remove(os.path.join(self.processed_path, 'train.csv'))

    def _unzip_test_db(self):
        """Unzip the raw db zip file."""
        print('Unzipping test database...')
        with zipfile.ZipFile(os.path.join(self.processed_path, 'test.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(self.processed_path, 'test'))

    def _csv_to_npy(self, file_name):
        """Import csv and save as npy."""
        # Import csv to npy array
        waveform = np.genfromtxt(os.path.join(self.processed_path, 'test', file_name),
                                 dtype=int, delimiter=',', skip_header=1)

        # Save to training test folder as npy file
        np.save(os.path.join(self.training_path, 'test', 'waveforms', file_name.split('.')[0] + '.npy'), waveform)

    def _import_train_data(self):
        """Import and format training data (train.csv)."""
        # Import training data as DataFrame
        data = pd.read_csv(os.path.join(self.processed_path, 'train.csv'))

        # Separate each column as npy array
        acoustic_data = data['acoustic_data'].values
        time_to_failure = data['time_to_failure'].values

        return acoustic_data, time_to_failure

    @staticmethod
    def _get_loading_cycles(time_to_failure):
        """Extract the start and end point of each loading cycle."""
        # Get peaks
        peaks = argrelmax(time_to_failure)[0]

        # Get troughs
        troughs = peaks - 1

        # Add start and end points
        peaks = np.concatenate(([0], peaks)).tolist()
        troughs = np.concatenate((troughs, [len(time_to_failure) - 1])).tolist()

        return [(peaks[idx], troughs[idx]) for idx in range(len(peaks))]

    def _save_loading_cycles(self, acoustic_data, time_to_failure, cycle_intervals):
        """Save loading cycles as npy files."""
        # Loop through intervals
        for idx, interval in enumerate(cycle_intervals):

            # Save to training test folder as npy file
            np.save(os.path.join(self.processed_path, 'acoustic_data_cycle_{}.npy'.format(idx + 1)),
                    acoustic_data[interval[0]:interval[1]])
            np.save(os.path.join(self.processed_path, 'time_to_failure_cycle_{}.npy'.format(idx + 1)),
                    time_to_failure[interval[0]:interval[1]])
