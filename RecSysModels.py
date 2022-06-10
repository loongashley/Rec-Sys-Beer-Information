#!/usr/bin/env python
# coding: utf-8

import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import turicreate as tc
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold,  cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.linear_model import SGDRegressor, RidgeCV, Ridge
from sklearn.metrics import mean_absolute_error, accuracy_score
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import surprise
from surprise import Reader, Dataset, KNNBasic, SVD, NMF,  KNNBaseline, KNNWithMeans, accuracy
from surprise.model_selection import cross_validate, train_test_split

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Reshape, Dot, Add, Activation, Lambda, Dense, Concatenate, Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers


from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

feature_path = "features.tsv"
tr_fpath = "train.tsv"
val_fpath = "val.tsv"
test_fpath = "test.tsv"
feature = pd.read_csv(feature_path, sep='\t', names = ['RowID','BrewerID','ABV','DayofWeek','Month','DayofMonth','Year','TimeOfDay','Gender','Birthday','Text','Lemmatized','POS_Tag'])
df_train = pd.read_csv(tr_fpath, sep='\t', names = ['RowID', 'BeerID', 'ReviewerID', 'BeerName', 'BeerType', 'Label'])
df_val = pd.read_csv(val_fpath, sep='\t', names = ['RowID', 'BeerID', 'ReviewerID', 'BeerName', 'BeerType', 'Label'])
df_test = pd.read_csv(test_fpath, sep='\t', names = ['RowID', 'BeerID', 'ReviewerID', 'BeerName', 'BeerType', 'Label'])

select_features = feature.iloc[:,[0,1,2,11]]
merge_tr_feat = pd.merge(df_train,select_features, on=['RowID'])
merge_test_feat = pd.merge(df_test,select_features, on=['RowID'])
merge_val_feat = pd.merge(df_val,select_features, on=['RowID'])


# setup the data for surprise

cleant = df_train.drop(['RowID','BeerName','BeerType'],axis=1)
cleanv = df_val.drop(['RowID','BeerName','BeerType'],axis=1)
reader = Reader(rating_scale=(0, 5))
surprise_data = Dataset.load_from_df(cleant[['BeerID','ReviewerID',
                                    'Label']],reader)

surprise_datav = Dataset.load_from_df(cleanv[['BeerID','ReviewerID',
                                     'Label']],reader)

surprise_datatest = Dataset.load_from_df(df_test[['BeerID','ReviewerID',
                                     'Label']],reader)

surprise_trainset = surprise_data.build_full_trainset()
NA,surprise_valset = train_test_split(surprise_datav, test_size=1.0, shuffle=False)
NA,surprise_testset = train_test_split(surprise_datatest, test_size=1.0, shuffle=False)


# setup the data for turicreate

datatr_header = df_train.to_csv("turi_train.csv", header=['RowID', 'BeerID', 'ReviewerID', 'BeerName', 'BeerType', 'Label'], index=False)
datatest_header = df_test.to_csv("turi_test.csv", header=['RowID', 'BeerID', 'ReviewerID', 'BeerName', 'BeerType',  'Label'], index=False)
datatr = tc.SFrame.read_csv('turi_train.csv')
datatest = tc.SFrame.read_csv('turi_test.csv')


'''
1st Model- KNN Surprise
'''

def kmean_prediction(trainset, testset, rowid):
    
#     param_grid = {'k': [50, 60, 70, 100, 150, 200]}

#     kgs = surprise.model_selection.GridSearchCV(KNNWithMeans, param_grid, measures=['rmse', 'mae'], cv=5)
#     kgs.fit(surprise_data)
    
#     print("Best RMSE: " ,gs.best_score['rmse'])
#     print("Best MAE: " ,gs.best_score['mae'])
    
#     print("Best PARAMS: " ,gs.best_params['rmse'])

#     knn_model = KNNWithMeans(**kgs.best_params['rmse'])

    # best params from gridsearchcv

    knn_model = KNNWithMeans(k=150) # best params

    
    model = knn_model.fit(trainset)
    test_pred = knn_model.test(testset) 

    row = df_test['RowID'].tolist()
    kmean_prediction = [i.est for i in test_pred]

    # save to dataframe
    
    lists = list(zip(row, kmean_prediction))
    kmean_df = pd.DataFrame(lists, columns = ['row', 'rating'])    
    kmean_df.to_csv("A3-1.tsv", sep="\t", header = False, index = False) 

    return kmean_df


'''
2nd Model- Turicreate
'''

def turi_predictions():
    
    # train on all values
    
    merged_tr_val = pd.concat([df_train, df_val], ignore_index=True, sort=False)
    merged_tr_val.to_csv("merged.csv", index=False)
    data_all = tc.SFrame.read_csv('merged.csv', verbose=False)

    # predict with factorization model

    ft_model = tc.factorization_recommender.create(data_all, user_id='ReviewerID', item_id='BeerID', target='Label', verbose=False)
    pred_df = datatest.remove_column('Label', inplace=False)
    turi_predictions = ft_model.predict(pred_df)
    
    turi_df = df_test['RowID'].to_frame()
    turi_df['Prediction'] = [i for i in turi_predictions] 
    
    # write to 2nd runfile
    
    turi_df.to_csv("A3-2.tsv", sep="\t", header = False, index = False) 

    return turi_df

'''
3rd Model- NN
'''

def nn_prediction():
    hidden_units = (32,4)
    beer_embedding_size = 8
    user_embedding_size = 8

    # Each instance will consist of two inputs: a single user id, and a single beer id
    
    user_id_input =Input(shape=(1,), name='ReviewerID')
    beer_id_input = Input(shape=(1,), name='Beer_ID')
    user_embedded = Embedding(df_train.ReviewerID.max()+1, user_embedding_size, 
                                           input_length=1, name='user_embedding')(user_id_input)
    beer_embedded = Embedding(df_train.BeerID.max()+1, beer_embedding_size, 
                                            input_length=1, name='beer_embedding')(beer_id_input)
    
    # concatenate embeddings and remove useless dim 
    
    concatenated = Concatenate()([user_embedded, beer_embedded])
    out = Flatten()(concatenated)

    # add hidden layers
    
    for n_hidden in hidden_units:
        out = Dense(n_hidden, activation='relu')(out)

    # single output: our predicted rating
    
    out = Dense(1, activation='linear', name='prediction')(out)

    model = Model(
        inputs = [user_id_input, beer_id_input],
        outputs = out,
    )
    model.summary(line_length=88)

    model.compile(
        tf.optimizers.Adam(0.01),
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=['mse','mae'],
    )

    history = model.fit(
        [df_train.ReviewerID, df_train.BeerID],
        df_train.Label,
        batch_size=5000,
        epochs=20,
        verbose=0,
        validation_split=.05,
    )
    
    test_user_id = df_test['ReviewerID']
    test_beer_id = df_test['BeerID']

    preds = model.predict([ test_user_id,
                            test_beer_id ])

    nn_df = df_test['RowID'].to_frame()
    nn_df['Prediction'] = [i for i in preds]
    nn_df['Prediction'] = nn_df['Prediction'].str[0]
    
    # save to tsv

    nn_df.to_csv("A3-3.tsv", sep="\t", header = False, index = False) 
    print("Neural Network DF: ",nn_df)

    return nn_df

'''
4th Model- SGD Regressor on text - BEST!!!
'''


def SGD_predictions(mergedtrain, mergedtest):

    # merge features with train and test dataset 
        
    select_features = feature.iloc[:,[0,1,2,11]]
    mergedtrain = pd.merge(df_train,select_features, on=['RowID'])
    mergedtest = pd.merge(df_test,select_features, on=['RowID'])
    
    # count vect for train dataset 
    list_corpus = mergedtrain["Lemmatized"].apply(lambda x: np.str_(x))

    count_vect = CountVectorizer(analyzer='word', token_pattern=r'[a-zA-Z]+')
    count_vect.fit_transform(list_corpus) 
    X = count_vect.transform(list_corpus)
    y = merge_tr_feat["Label"]

    # using SGD Regressor
    
    model = SGDRegressor()
    model.fit(X, y)

    # parameter tuning for SGD Regressor

#     cv = KFold(n_splits=5, random_state=1, shuffle=True) 
#     param_grid = { 'alpha': 10.0 ** -np.arange(1, 7),
#                     'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
#                     'penalty': ['l2', 'l1', 'elasticnet'],
#                     'learning_rate': ['constant', 'optimal', 'invscaling'],
#                     'max_iter' : [100, 1000]}
    
#     clf = RandomizedSearchCV(model, param_grid, scoring='neg_mean_absolute_error',cv=cv, verbose=True, n_jobs=-1, random_state=2)

#     print("Best score for SGDRegressor: " + str(clf.best_score_))
#     print("Best parameters for SGDRegressor: " + str(clf.best_params_))
    
    # best params
    
    sgd_params= {'penalty': 'l2', 'max_iter': 100, 'loss': 'epsilon_insensitive', 'learning_rate': 'invscaling', 'alpha': 1e-05}
    
    model = SGDRegressor(**sgd_params)
    model.fit(X, y)
    
    # count vect for test dataset 

    t_corpus = merge_test_feat["Lemmatized"].apply(lambda x: np.str_(x))
    t_test_bow = count_vect.transform(t_corpus)
    
    # prediction on test set

    SGD_pred = model.predict(t_test_bow)
    SGD_df = df_test['RowID'].to_frame()
    SGD_df['Prediction'] = [i for i in SGD_pred]
    
    # save to tsv
    
    SGD_df.to_csv("A3-4.tsv", sep="\t", header = False, index = False) 

    return SGD_df


'''
5th Model- Ridge
'''


def ridge_predictions(mergedtrain, mergedtest):
    
    # count vect for train dataset 
    
    list_corpus = mergedtrain["Lemmatized"].apply(lambda x: np.str_(x))
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'[a-zA-Z]+')
    count_vect.fit_transform(list_corpus) 
    X = count_vect.transform(list_corpus)
    y = mergedtrain["Label"]
    
    # count vect for test dataset 

    test_corpus = mergedtest["Lemmatized"].apply(lambda x: np.str_(x))
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'[a-zA-Z]+')
    count_vect.fit_transform(list_corpus) 
    X_test = count_vect.transform(test_corpus)


    model = Ridge(alpha=100, random_state=2)
    model.fit(X, y)
    
    # prediction on test set
    
    ridge_pred = model.predict(X_test)
    ridge_df = df_test['RowID'].to_frame()
    ridge_df['Prediction'] = [i for i in ridge_pred]
    
    # save to tsv
    
    ridge_df.to_csv("A3-5.tsv", sep="\t", header = False, index = False) 

    return ridge_df


if __name__ == "__main__":
    kmean_predictions = kmean_prediction(surprise_trainset, surprise_testset, df_test['RowID'])
    second_turi_predictions = turi_predictions()
    third_nn_predictions = nn_prediction()
    fourth_SGD_predictions = SGD_predictions(merge_tr_feat, merge_test_feat)
    fifth_Ridge_predictions = ridge_predictions(merge_tr_feat, merge_test_feat)