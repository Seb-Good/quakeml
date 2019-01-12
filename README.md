# QuakeML
Can you predict a laboratory earthquake?

## Data
Download the LANL Earthquake Prediction dataset (all.zip) from [Kaggle](https://www.kaggle.com/c/LANL-Earthquake-Prediction/data)
and place in the directory ```/data/raw/```.
```
├── data
│   ├── features
│   ├── processed
│   ├── raw
│   │   ├── all.zip
│   ├── training
├── quakeml
├── README
.getignore
LICENSE.txt
README.md
requirements.txt
setup.py
```
Run the notebook ```/quakeml/training/data/notebooks/create_db.ipynb``` to generate testing and training datasets.

````python
from quakeml.training.data.dataset import Dataset

# Initialize
datasets = Dataset()

# Generate dataset
datasets.generate_db()
````

## License
[MIT](LICENSE.txt)
