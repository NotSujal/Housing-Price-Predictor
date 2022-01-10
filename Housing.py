# coding: utf-8
print("Starting")
# Housing Price Calculation


print("Importing")
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Imputer (replaced with pipeline)
def impute(data):
    imputer = SimpleImputer(strategy='median')
    modified_data = imputer.fit_transform(data)
    return pd.DataFrame(modified_data,columns=data.columns)


# Correlations(generate a corelation data)
def correlate(data,lable):
    corr_matrix = data.corr()
    return corr_matrix[lable].sort_values(ascending=False)


# Train-Test Splitting (StratifiedShuffleSplit)
def train_test_split(data,lable):
    split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data,data[lable]):
        train_set = data.loc[train_index]
        test_set = data.loc[test_index]
    return train_set, test_set


# Cross-Validation
def cross_validate(model,data,scoring="neg_mean_squared_error",cv_folds=10):
    print("Cross Validating")
    from sklearn.model_selection import cross_val_score
    score = cross_val_score(model,data_np,labels,scoring=scoring, cv=cv_folds)
    score =  np.sqrt(-score)
    print(f"{type(model)}  : {score.mean()+ score.std()}")


#  Model Selection
def model_lr(data,labels):
    lr_model = LinearRegression()
    lr_model.fit(data,labels)
    a = cross_validate(lr_model,data)
    return lr_model
    
def model_dtr(data,labels):
    dtr_model = DecisionTreeRegressor()
    dtr_model.fit(data,labels)
    a = cross_validate(dtr_model,data)
    return dtr_model
    
def model_rfr(data,labels):
    rfr_model = RandomForestRegressor()
    rfr_model.fit(data,labels)
    a = cross_validate(rfr_model,data)
    return rfr_model
    
def search_model(data,labels):
    model_lr (data,labels)
    model_dtr(data,labels)
    model_rfr(data,labels)

# Saving Data
def savemodel(model, name):
    from joblib import dump
    dump(model,name)
    
    
# Pipelining(SimpleImputer ,StandardScaler)
pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy='median')),
    ("std_scalar",StandardScaler())
])

# Working
if __name__ == "__main__":
    print("Working")
    data = pd.read_csv("housing.csv")

    # Preprocessing data
    data_tr, test_set = train_test_split(data,"ocean_proximity")
    labels = data_tr["median_house_value"].copy()
    data_tr = data_tr.drop("median_house_value",axis=1)
    data_np = pipeline.fit_transform(data_tr)

    # Setting Up Model
    model = model_rfr(data_np,labels)

    # Testing
    cross_validate(model,test_set,cv_folds=5)

    #Saving Model
    savemodel(model,"median_house_value_rfr_predictor_err_63652USD_.joblib")
