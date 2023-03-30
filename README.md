# AI4PM_StressDetection

We use virtual environment to manage the dependencies.

1. `python -m venv env` to create virtual environment
2. `pip install -r requirements.txt` to install the dependencies
3. Unzip the `model.zip.001` and `model.zip.002`, and get `model.joblib`
4. `python ModelTrainer.py` to run the trainer (need to download the dataset first)
5. `python StressDetector.py` to detect stress

The dataset is from [here](https://www.kaggle.com/datasets/qiriro/swell-heart-rate-variability-hrv).