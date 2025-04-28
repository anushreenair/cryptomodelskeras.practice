# Cryptocurrency Price Prediction using LSTM
ijk
## Overview
This project implements a **Long Short-Term Memory (LSTM) neural network** to predict cryptocurrency prices based on historical data. The model is built using **TensorFlow/Keras**, processes data using **scikit-learn**, and visualizes results with **Matplotlib**.

## Features
- Fetches **real-time historical data** for a cryptocurrency (e.g., BTC-USD, ETH-USD) using **Yahoo Finance (yfinance)**.
- Preprocesses data using **MinMaxScaler** to normalize prices.
- Creates **sequences** of past price data to train the model.
- Implements an **LSTM-based neural network** with dropout layers for better generalization.
- Splits data into **training (80%) and testing (20%) sets**.
- Trains the model to predict future cryptocurrency prices.
- Plots **actual vs. predicted** prices to visualize model performance.
- Saves the trained model as `crypto_model.keras`.

## Installation
Ensure you have Python installed, then install the required dependencies:
```sh
pip install numpy pandas matplotlib seaborn tensorflow keras scikit-learn yfinance
```

## Usage
1. Run the script:
   ```sh
   python crypto_model_keras.py
   ```
2. Enter the cryptocurrency symbol (e.g., BTC-USD, ETH-USD, DOGE-USD).
3. The script will:
   - Fetch historical data
   - Train the LSTM model
   - Predict future prices
   - Display a graph comparing actual vs. predicted prices

## Model Architecture
The model consists of:
- **LSTM layers** with 50 units each
- **Dropout layers** (0.2 probability) to prevent overfitting
- **Dense layers** for final predictions
- **Adam optimizer** with Mean Squared Error (MSE) loss function

## Results
After training, the model generates a graph showing **actual vs. predicted prices**, helping assess prediction accuracy.

## Future Improvements
- Implement additional technical indicators for better accuracy.
- Experiment with **GRU or Transformer-based models**.

USED
- Optimize hyperparameters using **Grid Search or Bayesian Optimization**.
- Deploy the model as an **API for real-time price prediction**.

## License
This project is open-source and available for modification and distribution.

