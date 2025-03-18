# Stock Price Prediction and Clustering

## Overview
This project aims to predict the stock prices of Zomato using an LSTM (Long Short-Term Memory) neural network. Additionally, the project includes a clustering analysis on a dataset containing earnings and orders to identify patterns using the K-Means algorithm.

## Features
- Fetches stock price data from Yahoo Finance.
- Computes 100-day and 200-day moving averages.
- Prepares the data for training and testing.
- Trains an LSTM model to predict stock prices.
- Visualizes predicted vs. actual prices.
- Implements K-Means clustering on earnings data.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow/Keras
- Yahoo Finance API

## output
![image](https://github.com/user-attachments/assets/4eb1c3a6-6c41-4a80-9b81-0959be301c5e)
![image](https://github.com/user-attachments/assets/3e662723-8978-40d6-b585-6f8aa462b532)
![image](https://github.com/user-attachments/assets/46442058-e8bf-4dcf-b602-d78e25cac589)
![image](https://github.com/user-attachments/assets/216bb07c-e38c-45c0-be3e-ceacbc6960f6)




### Stock Price Prediction
1. Load historical stock data from Yahoo Finance.
2. Compute moving averages.
3. Normalize and prepare the data.
4. Train the LSTM model.
5. Predict and visualize stock prices.

### K-Means Clustering
1. Load the dataset (`totalearning.csv`).
2. Preprocess and normalize the data.
3. Apply K-Means clustering.
4. Visualize the clusters.

## Results
- Predicted vs. actual stock prices are plotted.
- Data is clustered based on total earnings and orders.

## Future Improvements
- Implement hyperparameter tuning for better LSTM performance.
- Use additional features for clustering analysis.
- Deploy the model for real-time predictions.

## Contributors
- **Avinash Kumar**

## License
This project is licensed under the MIT License.

