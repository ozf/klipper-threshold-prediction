# klipper-threshold-prediction

This project predicts optimal threshold values for 3D printing using the Klipper firmware based on various input parameters.

## Features

- Train a model to predict threshold values.
- Predict threshold values based on user input.

## Files

- `klipper_training_data.csv`: Dataset used for training the model.
- `train_model.py`: Script to train the model.
- `predict_threshold.py`: Script to predict threshold values.
- `requirements.txt`: Required Python packages.

## Usage

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. Train the Model

```sh
python train_model.py
```

### 3. Predict Threshold

```sh
python predict_threshold.py
```
