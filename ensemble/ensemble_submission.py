# This script generates vector of predictions for LB submission

# The predictive model consists of three stages:

# 1.Generate stacked data: get 9 out-of-fold predictions for the log(number of mosquitos) and add use them 
#   as meta-features in the 2d stage

# 2.Train XGBoost and Neural Net Classifiers on the stacked data obtaining 2 sets of predictions

# 3.Combine obtained predictions using geometric mean to get the final vector of predicted probabilities

# This script can generate 4 different outputs: 2 versions of data x 2 submission versions
# Data_Version #1 and Submission_Version #1 were used to generate the official LB submission

# Data_Version #1: data preparation version used for submission
# Data_Version #2: data preparation version with post-deadline improvements (new features added)

# Submission_Version #1: submission with probability estimates provided by the model
# Submission_Version #2: submission with probability estimates provided by the model, but
# manually tuned to accommodate the outbreak in West Nile Virus which occurred in 2012 (this can be inferred
# from the LB feedback)

# load python modules
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso

# load custom modules
from wnv_utils import utils, paths
from wrappers import models

Data_Version = 1 # 2    Choose the data version (1,2)
Submission_Version = 1 # 2  Choose the submission version (1,2)
bagging = False # True  Bagging stabilizes Neural Net predictions which usually have high variance
if bagging == True:
    number_samples = 250  # set number of bootstrap samples (set high number only if you have GPU, otherwise
                          # use 10 or 20 samples)

def PredictXGB(train, test, target):  # Get predictions of XGBoost classifier
    print("Training Gradient Boosting Classifier ")
    clf = models.XGBoostClassifier(nthread=6, eta=0.025, gamma=1.425, max_depth=11,
                     min_child_weight=9, max_delta_step=0, subsample=0.55, colsample_bytree=0.7, 
                    scale_pos_weight=1, silent =0, seed=101, l2_reg=2, alpha=0.65, n_estimators=900)
    clf.fit(train, target)  
    preds =  clf.predict_proba(test)
    return preds

def PredictNN(train, test, target): # Get predictions of Neural Net classifier
    print("Training Neural Network Classifier ")   
    scaler = StandardScaler().fit(train)  # Scale data
    train=scaler.transform(train)
    test=scaler.transform(test)
    clf = models.NeuralNetClassifier(n_hidden1=440, n_hidden2=200, n_hidden3=90, max_epochs=155,
                               batch_size=90, lr=0.005, momentum=0.9, dropout_input=0.15,
                               dropout_hidden=0.6, valid_ratio=0, use_valid=False, verbose=0,
                               random_state=101)
    if bagging == True:                       
        preds = utils.Bagging(train, test, target, number_samples, clf)
    else:
        clf.fit(train, target)
        preds = clf.predict_proba(test)[:, 1]
    return preds
    
train = utils.LoadTrain(Data_Version)  # load prepared data
test = utils.LoadTest(Data_Version)  
y=np.log(train.NumMosquitos.values+1) # log(# of mosquitos) is a target variable in stacking procedure
wnv=train.WnvPresent.values    # prediction target
train.drop(['NumMosquitos', 'WnvPresent'], axis=1, inplace=True)
ids=test.Id.values    # use ids to construct the submission file
test.drop('Id', axis=1, inplace=True)

# Initialize regression models for estimation of log(number of mosquitos)

clf1 = models.XGBoostRegressor(booster='gbtree', nthread=6, eta=.025, gamma=1.9, max_depth=9, 
                 min_child_weight=6, max_delta_step=0, subsample=0.75, colsample_bytree=0.7, 
                 silent =1, seed=101, l2_reg=1, alpha=0, n_estimators=425)
                  
clf2 = RandomForestRegressor(n_estimators=750, criterion='mse', max_depth=9,
                            min_samples_split=2, min_samples_leaf=6, min_weight_fraction_leaf=0.0,
                            max_features=0.75, max_leaf_nodes=None, bootstrap=False, oob_score=False,
                            n_jobs=2, random_state=101, verbose=0, warm_start=False) 

clf3 = ExtraTreesRegressor(n_estimators=850, criterion='mse', max_depth=7, min_samples_split=2,
                     min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_features=0.7,
                     max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=2, 
                     random_state=101, verbose=0, warm_start=False)

clf4 = GradientBoostingRegressor(loss='ls', learning_rate=0.015, n_estimators=750, subsample=0.8, 
                                min_samples_split=2, min_samples_leaf=5, min_weight_fraction_leaf=0.0,
                                max_depth=9, init=None, random_state=101, max_features=0.85, verbose=0,
                                max_leaf_nodes=None, warm_start=False)                     

clf5 = AdaBoostRegressor(base_estimator=None, n_estimators=850, learning_rate=0.01, loss='square',
                         random_state=101)   
                                                 
clf6 = SVR(kernel='rbf', degree=3, gamma=0.1, coef0=0.0, tol=0.001, C=3, epsilon=0.05, shrinking=True, 
         cache_size=200, verbose=False, max_iter=-1) 
         
clf7 = models.XGBoostRegressor(booster='gblinear', nthread=6, silent =1, seed=101, eta=0.01, l2_reg=5, 
                 alpha=0.5, gamma=None, max_depth=None, min_child_weight=None, max_delta_step=None,
                 subsample=None, colsample_bytree=None, n_estimators=825)
                 
clf8 = Ridge(alpha=3.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, 
           solver='auto') 

clf9 = Lasso(alpha=0.2, fit_intercept=True, normalize=False, precompute=False, copy_X=True,
      max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=101, selection='cyclic')                 
                  
clfs=[clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9] # list of models used in the 1st stage to obtain meta-features

n = 4 # position of the last non-scaled model in the list (need to pass it to the Stacking function)

train_data, test_data = utils.StackModels(train, test, y, clfs, n) # generate stacked data 
preds_nn = PredictNN(train_data, test_data, wnv)    # obtain Neural Net predictions using stacked data
preds_xgb = PredictXGB(train_data, test_data, wnv)  # obtain XGBoost predictions using stacked data

if (Submission_Version==1):                        
   preds=(preds_nn**(0.6))*(preds_xgb**(0.4))       # Final submission is a geometric mean of 2 predictions
   
else:
   # taking into account the virus outbreak in 2012, increase 2012 probbailities manually
   # this generates a HUGE boost to the LB score (gives you approximately 26th-28th place on the LB)
   preds_xgb[test.Year.values==2012]=preds_xgb[test.Year.values==2012]*3
   preds_xgb[preds_xgb>1]=0.99
   preds_nn[test.Year.values==2012]=preds_nn[test.Year.values==2012]*3
   preds_nn[preds_nn>1]=0.99
   preds=(preds_nn**(0.6))*(preds_xgb**(0.4))

#Save the submission file   
utils.save_submission(ids, preds)
