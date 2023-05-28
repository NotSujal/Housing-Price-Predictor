# Final Testing
from joblib import load
import pandas as pd
from random import randint
import datetime

# not creating a new pipeline instead using from Housing.py

# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# pipeline = Pipeline([
#     ("imputer",SimpleImputer(strategy='median')),
#     ("std_scalar",StandardScaler())
# ])
from Housing import pipeline

data = pd.read_csv("../data/housing.csv")

loaded_mdl = load("../data/median_house_value_rfr_predictor_.joblib")

rng =randint(0,len(data)-50)
somedata = data[rng:rng+50]

somedata_X = somedata.drop("median_house_value",axis=1)
somedata_tr_X = pipeline.fit_transform(somedata_X)
somedata_Y = somedata["median_house_value"].copy()
predictions = loaded_mdl.predict(somedata_tr_X)

with open("../data/test_data.txt","w") as f:
    f.write("")

test_data = f"{datetime.datetime.now()}\n\n"
for i in range(len(predictions)):
    test_data += f"[{int(somedata_Y.iloc[i] - predictions[i]) }\t]\t\t Actual: {somedata_Y.iloc[i]} \t\tPredicted: {predictions[i]}\n"
    test_data += "\n\n"



with open("../data/test_data.txt","a") as f:
    f.write(test_data)