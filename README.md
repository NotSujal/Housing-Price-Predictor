# Housing Predictor

The Housing Predictor is a powerful model designed to accurately predict the median house value based on various features of the housing block. This model utilizes advanced machine learning techniques to provide precise estimates, helping individuals and organizations make informed decisions in the real estate market.

## Getting Started

To create the model locally, simply run the `Housing.py` file. The model file is not included in this repository due to its large size. Running the script will train the model on the provided dataset and generate the necessary files for prediction.

## Dataset Description

The dataset used for training the model consists of several key features that contribute to the prediction of median house value. Here are some details about the features:

- `housingMedianAge`: Median age of a house within a block; a lower number is a newer building.
- `totalRooms`: Total number of rooms within a block.
- `totalBedrooms`: Total number of bedrooms within a block.
- `population`: Total number of people residing within a block.
- `households`: Total number of households, a group of people residing within a home unit, for a block.
- `medianIncome`: Median income for households within a block of houses (measured in tens of thousands of US Dollars).
- `medianHouseValue`: Median house value for households within a block (measured in US Dollars).
- `oceanProximity`: Location of the house w.r.t ocean/sea.

## Ocean Proximity

The `oceanProximity` feature provides information about the location of the house in relation to the ocean or sea. Different proximity levels are represented by specific codes. The chosen codes were based on a thorough analysis of the dataset and their correlation with the median house value.

- `NEAR BAY`: Code 90
- `NEAR OCEAN`: Code 90
- `<1H OCEAN`: Code 80
- `INLAND`: Code 40
- `ISLAND`: Code 50

These codes were selected because they demonstrated a strong correlation with the median house value. Extensive analysis and feature engineering techniques were applied to identify the most influential factors in predicting the house value accurately.

## Model Evaluation

The Housing Predictor model has been evaluated using different regression algorithms. Here are the error scores obtained:

- Linear Regression:
    - Mean Error: 7.05e4
    - Standard Deviation: 2.22e3

- Decision Tree Regressor:
    - Mean Error: 8.43e4
    - Standard Deviation: 1.77e3

- Random Forest Regressor (Selected Model):
    - Mean Error: 6.16e4
    - Standard Deviation: 1.39e3

The models were trained and evaluated using cross-validation techniques to assess their performance. The Random Forest Regressor was chosen as the final model because it demonstrated the lowest mean error score and standard deviation. This model takes into account the complex interactions between different features, resulting in highly accurate predictions of median house values.

We encourage you to leverage the power of the Housing Predictor model to make informed decisions in the real estate market.
