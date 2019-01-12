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
        print('Complete!\n')

    def _unzip_db(self):
        """Unzip the raw db zip file."""
        print('Unzipping database...')
        with zipfile.ZipFile(os.path.join(self.raw_path, 'all.zip'), 'r') as zip_ref:
            zip_ref.extractall(self.processed_path)

    def _unzip_test_db(self):
        """Unzip the raw db zip file."""
        print('Unzipping test database...')
        with zipfile.ZipFile(os.path.join(self.processed_path, 'test.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(self.processed_path, 'test'))

    def _get_test_db(self):
        """Create the test dataset and convert from csv to npy files."""
        print('Creating test database...')
        # Unzip test dataset
        self._unzip_test_db()

        # Get list of files
        file_names = os.listdir(os.path.join(self.processed_path, 'test'))

        # Move test files to training folder
        _ = Parallel(n_jobs=-1)(delayed(self._save_npy)(file_name) for file_name in file_names[0:10])

        # Remove test directory
        shutil.rmtree(os.path.join(self.processed_path, 'test'))

        # Remove zip file
        os.remove(os.path.join(self.processed_path, 'test.zip'))

    def _save_npy(self, file_name):
        """Import csv and save as npy."""
        # Import csv to npy array
        waveform = np.genfromtxt(os.path.join(self.processed_path, 'test', file_name),
                                 dtype=int, delimiter=',', skip_header=1)

        # Save to training test folder as npy file
        np.save(os.path.join(self.training_path, 'test', 'waveforms', file_name.split('.')[0] + '.npy'), waveform)
