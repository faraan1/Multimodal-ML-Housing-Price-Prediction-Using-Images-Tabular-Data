# Multimodal House Price Prediction
This project demonstrates how to build a multimodal deep learning model using PyTorch to predict house prices. The model leverages two types of data: tabular features (e.g., bedrooms, bathrooms, square footage) and simulated image features, combining them to make a final prediction.

## Features
- Data Generation: Automatically generates a sample `housing_data.csv` file if one doesn't exist, which includes tabular data and dummy image paths.
- Simulated Image Features: Uses a `SimulatedCNNFeatureExtractor` to create random feature vectors, standing in for features that would be extracted by a real Convolutional Neural Network (CNN) from house images.
- Data Preprocessing: Standardizes tabular features using `StandardScaler` from scikit-learn.
- Multimodal Architecture: A simple feed-forward neural network is defined in PyTorch (`MultimodalHousePriceModel`) to process the concatenated tabular and image features.
- Training & Evaluation: The model is trained using `Adam` optimizer and `MSELoss`. It is then evaluated on a validation set using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to measure performance.
- Sample Prediction: Provides a sample prediction to demonstrate the model's output and calculate the error against the actual price.

## Requirements
The code is written in Python and requires the following libraries:
- `pandas`
- `numpy`
- `torch`
- `scikit-learn`

## Code Structure
- `create_sample_csv()`: Function to generate a synthetic dataset for demonstration purposes.
- `SimulatedCNNFeatureExtractor`: A class that simulates a CNN's role in extracting features from images.
- `HousingDataset`: A custom PyTorch `Dataset` class to handle the combined features and targets.
- `MultimodalHousePriceModel`: The PyTorch `nn.Module` defining the neural network architecture.

## Training Loop 
The code block for training the model over a set number of epochs.

## Evaluation
A section to calculate and display key performance metrics (MAE, RMSE) on a validation set.
