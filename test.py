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

data = pd.read_csv("housing.csv")

loaded_mdl = load("median_house_value_rfr_predictor_err_63652USD_.joblib")

rng =randint(0,len(data)-50)
somedata = data[rng:rng+50]

somedata_X = somedata.drop("median_house_value",axis=1)
somedata_tr_X = pipeline.fit_transform(somedata_X)
somedata_Y = somedata["median_house_value"].copy()
predictions = loaded_mdl.predict(somedata_tr_X)

with open("test_data.txt","w") as f:
    f.write("")

test_data = f"{datetime.datetime.now()}\n\n"
for i in range(len(predictions)):
    test_data += f"\n\n{rng+i}:Predicted:   {predictions[i:i+1]},   Original:   {list(somedata_Y[i:i+1])}"
    test_data += f"\nDifference:   {predictions[i:i+1] - list(somedata_Y[i:i+1])} "


with open("test_data.txt","a") as f:
    f.write(test_data)