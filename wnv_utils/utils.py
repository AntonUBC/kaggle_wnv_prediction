# This script contains functions used for data loading, cleaning, merging, feature engineering,
# and saving predictions
# It also contains a stacking function, used to obtain meta-features: predicted log(number of mosquitos) for the 2d stage
# Finally, there is a bagging (bootstrap) function designed to stabilize predictions of the Neural Net
# classifier

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, asin
from sklearn.preprocessing import StandardScaler
from wnv_utils import paths

path_train = paths.DATA_TRAIN_PATH
path_test=paths.DATA_TEST_PATH
path_weather=paths.DATA_WEATHER_PATH
path_sample_submission=paths.DATA_SUBMISSION_PATH

def buildLaggedFeatures(df, lag):  # df - dataframe, lag - list of numbers defining lagged values. Builds lagged weather features 
    new_dict={}
    for col_name in df:
        new_dict[col_name]=df[col_name]
        # create lagged Series
        for l in lag:
            if col_name!='Date' and col_name!='Station':
                new_dict['%s_lag%d' %(col_name,l)]=df[col_name].shift(l)
    res=pd.DataFrame(new_dict,index=df.index)
    return res

def DuplicatedRows(df): # Calculates number of duplicated rows by Date, Trap, Species
    grouped = df.groupby(['Date', 'Trap', 'Species'])
    num=grouped.count().Latitude.to_dict()
    df['N_Dupl']=-999
    for idx in df.index:
        d = df.loc[idx, 'Date']
        t = df.loc[idx, 'Trap']
        s = df.loc[idx, 'Species']
        df.loc[idx, 'N_Dupl'] = num[(d, t, s)]
    return df
    
def haversine(lat1, lon1, lat2, lon2): # Calculates the haversine distance between two Lat, Long pairs
    R = 6372.8 # Earth radius in kilometers
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    a = np.power(np.sin(dLat/2), 2) + np.multiply(np.cos(lat1), np.multiply(np.cos(lat2), np.power(np.sin(dLon/2), 2)))
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def ClosestStation(df): # identify the closest weather station
    df['lat1']=41.995   # latitude of 1st station
    df['lat2']=41.786   # latitude of 2d station
    df['lon1']=-87.933  # longitude of 1st station
    df['lon2']=-87.752  # longitude of 2d station
    df['dist1'] = haversine(df.Latitude.values, df.Longitude.values, df.lat1.values, df.lon1.values) #calculate distance
    df['dist2'] = haversine(df.Latitude.values, df.Longitude.values, df.lat2.values, df.lon2.values)
    indicator = np.less_equal(df.dist1.values, df.dist2.values) # determine which station is the closest
    st = np.ones(df.shape[0])
    st[indicator==0]=2
    df['Station']=st    # obtain station identifier for each row 
    df.drop(['dist1', 'dist2', 'lat1', 'lat2', 'lon1', 'lon2' ], axis=1, inplace=True)
    return df
    
def DateSplit(df):   # parse date in text format to get 3 separate columns for year, month, day
    df = df.copy()
    df['Year'] = pd.DatetimeIndex(df['Date']).year
    df['Month'] = pd.DatetimeIndex(df['Date']).month
    df['Day'] = pd.DatetimeIndex(df['Date']).day
    return df   
    
def GetDummies(df):   # Constructs dummy indicators for species
    dummies=pd.get_dummies(df['Species'])
    df = pd.concat([df, dummies], axis=1)
    return df  
    
def ExtendWnv(df):  # assigns WNV indicator to duplicate rows (if WNV=1, assign 1 to all duplicate rows)
    grouped = df.groupby(by=['Date', 'Trap', 'Species'], as_index=False)['WnvPresent'].max() 
    df.drop('WnvPresent', axis=1, inplace=True)
    grouped.columns = ['Date', 'Trap', 'Species', 'WnvPresent']
    result = df.merge(grouped, on=['Date', 'Trap', 'Species'], how="left") #.reset_index()
    return result

def MergeWeather(df1, df2):
    result = df1.merge(df2, on=['Date', 'Station'], how="left",  left_index=True)
    return result 
    
def LoadWeather():  # Load and prepare weather data
    days = [1, 3, 5, 8, 12]     #lagged weather values used as features
    weather = pd.read_csv(path_weather)
    weather.sort(['Date', 'Station'], axis=0, ascending=True, inplace=True)
    filter_out = ['Heat', 'CodeSum', 'Depth', 'Water1', 'SnowFall', 'StnPressure',  'SeaLevel', 'AvgSpeed' ]
    weather.drop(filter_out, axis=1, inplace=True) # these variables are not used in the analysis
    weather.replace(['  T','M','-'], [0.001, np.nan, np.nan], inplace=True) # replace "Trace" with 0.001, replace M and missing with NaN
    weather.WetBulb.fillna(method='bfill', inplace=True)  # replace missing WetBulb of 1st station with the value of 2d station
    weather.fillna(method='pad', inplace=True)   # replace all missing values of 2d station with values of 1st station
    weather1 = buildLaggedFeatures(weather[weather['Station']==1], days) # build lagged features for 1st station
    weather2 = buildLaggedFeatures(weather[weather['Station']==2], days) # build lagged features for 2d station
    weather = weather1.append(weather2)                                  # append data from 2 stations
    weather.sort(['Date', 'Station'], axis=0, ascending=True, inplace=True)
    return weather
    
def LoadTest(version): # Load and prepare test data
    print("Loading Test Data")
    test = pd.read_csv(path_test)
    test = DateSplit(test)
    test = DuplicatedRows(test)
    if version == 2: # calculate duplicated rows for various subgroups of data
         l = [['Date', 'Species'], ['Date', 'Trap'], ['Date', 'Block'], ['Year', 'Species'],
              ['Year', 'Trap'], ['Year', 'Block'], ['Month', 'Species'], ['Month', 'Trap'], 
              ['Month', 'Block']]
         for i in range(len(l)):
             test = AddMoreFeatures(test, l[i]) # compute number of duplicated rows for each group
    test = ClosestStation(test)
    weather = LoadWeather()
    test = MergeWeather(test, weather)
    test.replace(['UNSPECIFIED CULEX'], ['CULEX PIPIENS'], inplace=True) # replace Unspecified species with PIPIENS 
    test = GetDummies(test)
    filter_out = ['Address', 'Block', 'Street', 'Trap', 'AddressNumberAndStreet', 'AddressAccuracy',
                 'Date', 'Species', 'Station' ]
    test.drop(filter_out, axis=1, inplace=True) # these features are not used in prediction
    return test
    
def LoadTrain(version): # Load and prepare train data
    print("Loading Train Data")
    train = pd.read_csv(path_train)
    train = DateSplit(train)
    train = DuplicatedRows(train)
    if version == 2: # calculate duplicated rows for various subgroups of data
         l = [['Date', 'Species'], ['Date', 'Trap'], ['Date', 'Block'], ['Year', 'Species'],
              ['Year', 'Trap'], ['Year', 'Block'], ['Month', 'Species'], ['Month', 'Trap'],
              ['Month', 'Block']]
         for i in range(len(l)):
             train = AddMoreFeatures(train, l[i]) # compute number of duplicated rows for each group
    train = ExtendWnv(train)
    train = ClosestStation(train)
    weather = LoadWeather()
    train = MergeWeather(train, weather)
    train = GetDummies(train)
    filter_out = ['Address', 'Block', 'Street', 'Trap', 'AddressNumberAndStreet', 'AddressAccuracy',
                 'Date', 'Species', 'Station']
    train.drop(filter_out, axis=1, inplace=True) # these features are not used in prediction
    return train

# Note: This is a POST-DEADLINE improvement 
# This procedure calculates the number of duplicated rows by ['Date', 'Species'], ['Date', 'Trap'], ['Date', 'Block'],
# ['Year', 'Species'], and etc. This improves LB score by about 0.017-0.018
def AddMoreFeatures(df, list_cols):
    indicator = np.not_equal(df.N_Dupl.values, np.ones(df.shape[0]))
    df['N'] = indicator
    grouped = df.groupby(by=list_cols, as_index=False)['N'].sum() 
    grouped.columns = [list_cols[0], list_cols[1], 'N_Dupl_%s_%s' %(list_cols[0], list_cols[1])]
    df.drop('N', axis=1, inplace=True)
    result = df.merge(grouped, on=list_cols, how="left")   
    return result 

def save_submission(ids, predictions):
    submission = pd.DataFrame({"Id": ids, "WnvPresent": predictions})
    submission.to_csv(path_sample_submission, index=False) 

def StackModels(train, test, y, clfs, n): # train data (pd data frame), test data (pd date frame), Target data, List of clfs to stack, position of last non-scailed model in clfs. 

# StackModels() performs Stacked Aggregation on data: it uses 9 different models to get out-of-fold 
# predictions of log(number of mosquitos) for train data. It uses the whole train dataset to obtain 
# predictions for test. This procedure adds 9 meta-features (predictions of 9 models) to both train and 
# test data. Furthermore, since some models (e.g., SVR) require data to be scailed for better performance,
# there is a parameter to be passed to StackModels() which determines the position of the LAST non-scailed
# model in the list of estimators. That is, the model list (clfs) should be constructed in such way that 
# all non-scailed models are placed first in the list followed by scailed models. Consequently, parameter 
# n in StackModels() determines the position of the last non-scailed model in the model list.
    print("Generating Meta-features")
    training = train.as_matrix()
    testing = test.as_matrix()
    blend_train = np.zeros((training.shape[0], len(clfs))) # Number of training data x Number of classifiers
    blend_test = np.zeros((testing.shape[0], len(clfs)))   # Number of testing data x Number of classifiers
    years = np.unique(train.Year.values)   # years are used as folders for getting out-ot-fold predictions
    for j, clf in enumerate(clfs):
        print ('Training regressor [%s]' % (j))
        for i in range(len(years)):
           print ('Fold [%s]' % (i))
            
           # This is the training and validation set (train on 3 years, predict on the 4th year)
           X_tr = training[train.Year.values!=years[i]]
           Y_tr = y[train.Year.values!=years[i]]
           X_cv = training[train.Year.values==years[i]]
           scaler=StandardScaler().fit(X_tr)   # scale data for SVM and linear models                          
           X_tr_scale=scaler.transform(X_tr)
           X_cv_scale=scaler.transform(X_cv)
           
           if j<=n: # these models do not require scailing                                           
               clf.fit(X_tr, Y_tr)
               blend_train[train.Year.values==years[i], j] = clf.predict(X_cv)
           else:    # these models DO require scailing 
               clf.fit(X_tr_scale, Y_tr)
               blend_train[train.Year.values==years[i], j] = clf.predict(X_cv_scale)
               
        scaler=StandardScaler().fit(training)
        X_train_scale=scaler.transform(training)
        X_test_scale=scaler.transform(testing)   
        if j<=n:                                             
               clf.fit(training, y)
               blend_test[:, j] = clf.predict(testing)
        else:
               clf.fit(X_train_scale, y)
               blend_test[:, j] = clf.predict(X_test_scale)
       
    X_train_blend_full=np.concatenate((training, blend_train), axis=1) # add obtained 6 columns of predictions 
    X_test_blend_full=np.concatenate((testing, blend_test), axis=1)    # to both train and test data
    return X_train_blend_full, X_test_blend_full 

def Bagging(train, test, target, bagging_size, clf):
    
# This procedure performs bagging to stabilize predictions of the Neural Net Classifier
    
    rng = np.random.RandomState(1014)   # set random seed for bagging              
    num_train = train.shape[0]
    preds_bagging = np.zeros((test.shape[0], bagging_size), dtype=float)
    for n in range(bagging_size):
        sampleSize = int(num_train)
        index_base = rng.randint(num_train, size=sampleSize) # get random indices with replacement
        train_boot=train[index_base]
        y_boot=target[index_base]
        clf.fit(train_boot, y_boot)
        preds_bagging[:,n] = clf.predict_proba(test)[:, 1]
    preds_bagging = np.mean(preds_bagging, axis=1)  # compute average of bootstrap predictions
    return preds_bagging           
