import pandas as pd
from xgboost import XGBRegressor

from sklearn.preprocessing import LabelBinarizer, Imputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error as sk_mae
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.model_selection import cross_val_score, KFold

import numpy as np

# Impute function
def impute_data(x, y):

    cols_with_na = [col for col in x.columns
                    if x[col].isnull().any()]

    for col in cols_with_na:
        x[col + "_was_missing"] = x[col].isnull()
        y[col + "_was_missing"] = y[col].isnull()

    my_imputer = Imputer()
    x = my_imputer.fit_transform(x)
    y = my_imputer.transform(y)

    return (x, y)

# Prep predictor data
def prep_model(train_x, test_x):

    train_x = pd.get_dummies(train_x)
    test_x = pd.get_dummies(test_x)
    train_x, test_x = train_x.align(test_x, join='left', axis=1)

    train_x, test_x = impute_data(train_x, test_x)

    return (train_x, test_x)

# Train model and predict
def fit_xgb(train_x, train_y, test_x):
    
    xgb_model.fit(train_x, train_y, 
                  early_stopping_rounds = 10,
                  eval_set=[(train_x, train_y)], 
                  verbose = False)

    output = xgb_model.predict(test_x)

    return output


# VARIABLES:
live = False

# Load files
file_path = 'input/'
train_data = pd.read_csv(file_path + 'train.csv')

train_x = train_data.drop('SalePrice', axis=1)
train_y = train_data.SalePrice

if live==False:
    
    cv_n = 10
    cv_n_running = 0
    print("Running iteration: ", end=" ")
    kf = KFold(n_splits=cv_n, shuffle=True, random_state=1)
    score = np.empty(cv_n, dtype=float)
    for train_index, test_index in kf.split(train_x):

        train_x_cv, test_x_cv = train_x.iloc[train_index,:], train_x.iloc[test_index,:]
        train_y_cv, test_y_cv = train_y.iloc[train_index], train_y.iloc[test_index]
        
        train_x_cv, test_x_cv = prep_model(train_x_cv, test_x_cv)
        
        my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
        
        my_model.fit(train_x_cv, train_y_cv, 
                      early_stopping_rounds = 10,
                      eval_set=[(train_x_cv, train_y_cv)], 
                      verbose = False)
        
        output = my_model.predict(test_x_cv)
        
        error = sk_mae(test_y_cv, output)
        print("{}%" .format((cv_n_running+1)/cv_n*100), end=" ")
        
        score[cv_n_running] = error
        cv_n_running += 1
        
        #train_cat_cols = train_x.select_dtypes(include=['object'])
    print(np.mean(score))
    
else:
    test_data = pd.read_csv(file_path + 'test.csv')
    test_x = test_data.copy()
    
    train_x, test_x = prep_model(train_x, test_x)
    
    output = fit_xgb(train_x, train_y, test_x)
    

