#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 01:02:28 2017

@authors: Carla Agurto, Pablo Polosecki

This file calls functions of polyssifier, but incorporates more general metrics
for model assessment, a standalone function to run a classifier and the use of
pandas instead of mat files. Binary and continuous predictions are stored in a
csv file when necessary, in addition to a summary of the results
 
"""

import polyssifier as ps
import logging
from sklearn.model_selection import GroupKFold
from os import path
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score 
from sklearn.metrics import roc_auc_score, recall_score, roc_curve


def evaluation_metrics(table, table2, target):
    
    list_topk = table.columns.tolist()
    list_topk.remove(target)
    list_topk.remove('Classified correctly %')
    
    #initialize table to save metrics
    metrics = ['Accuracy', 'Error', 'Precision', 'Recall', 'Sensitivity', 
               'Specificity', 'AUC', ' brier_score_loss']
    results = pd.DataFrame(index = metrics , columns = list_topk )
    
    for topk in list_topk:
        results.loc[metrics[0]][topk] = accuracy_score(table[target].astype(float).values,
                    table[topk].astype(float).values)
        results.loc[metrics[1]][topk] = 1 - results.loc[metrics[0]][topk]
        results.loc[metrics[2]][topk] = precision_score(table[target].astype(float).values,
                    table[topk].astype(float).values)
        results.loc[metrics[3]][topk] = recall_score(table[target].astype(float).values,
                    table[topk].astype(float).values)
        fpr, tpr, thresh = roc_curve(table[target].astype(float).values,
                    table[topk].astype(float).values)
        results.loc[metrics[4]][topk] = tpr[1] #sensitivity in index 1
        results.loc[metrics[5]][topk] = 1-fpr[1] #specificity in index 1
        try:
            #For probabilities copute the AUC and the brier score
           results.loc[metrics[6]][topk] = roc_auc_score(table2[target].astype(float).values,
                    table2[topk].astype(float).values)
           results.loc[metrics[7]][topk] = brier_score_loss(table2[target].astype(float).values,
                    table2[topk].astype(float).values)
        except:
            print('There are not probabilities for this classifier')
        
    return results


def rightclassrate(row):
    
#to calculate the percentage that the sample was correctly classified 
    
    rate = row.tolist().count(row[-1])-1 #one since we include the label at the end
    rate = rate/0.01/(len(row)-1)
    return rate


def Classifiers(dirfilename,topkfeatures,classifiers_name, runs_per_subj, 
                table, target,list_features, subjects, fix_features, list_fix_features,
                fix_train_test, list_train, list_test, kfold,
                covariate_detrend_params=None, randomvalue=0):


    compute_results = True
    parallel_flag = False    
    
    #Loading data / create directory to store the results    
    out_dir = dirfilename
    if not os.path.isdir(out_dir): #remove the extension
        os.makedirs(out_dir)
   
    data = table[list_features].as_matrix()
    labels = np.array(table[target])  
    subject_labels = np.array(table[subjects])    
    
    #-------------------------------------------------
    # Initializing logger to write to file and stdout
    #-------------------------------------------------
    logging.basicConfig(format="[%(asctime)s:%(module)s:%(levelname)s]:%(message)s",
                        filename=path.join(out_dir, 'log.log'),
                        filemode='w',
                        level=logging.DEBUG)
    logger = logging.getLogger()
    ch = logging.StreamHandler(logging.sys.stdout)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    #--------------------------------------------
    # Cleaning data and labels (removing NaNs)
    #--------------------------------------------
    logging.info("Trying to load data")
    if np.any(np.isnan(data)):
        h=np.nonzero(np.isnan(data))
        data[h[0],h[1]]=0
        logging.warning('Nan values were removed from data')
    if np.any(np.isinf(data)):
        h=np.nonzero(np.isinf(data))
        data[h[0],h[1]]=0
        logging.warning('Inf values were removed from data')
               
    ksplit = kfold
    logging.info("Folds are {}".format(ksplit))
    logging.info("Test subjects per split:{}".format(data.shape[0]/(ksplit*runs_per_subj)))

    #-----------------------------
    #CLASSIFIER AND PARAM DICTS
    #-----------------------------
    classifiers, params = ps.make_classifiers(classifiers_name)
 
    # Make subject-wise folds
    group_kfold = GroupKFold(n_splits=ksplit)  

    split_gen = list(group_kfold.split(data, labels, subject_labels))
    # Extract the training and testing indices from the k-fold object,
    # which stores fold pairs of indices.
    fold_pairs = [(tr, ts) for (tr, ts) in split_gen]
    assert len(fold_pairs) == ksplit
    
    #This code is to save in a csv files the fold_pairs
    #--------------------------------------------------
    
    dict_feat = dict(zip(range(data.shape[0]), subject_labels[list(set(range(data.shape[0])))]))
    fold_pairs_table = pd.DataFrame(index = ['sample_' + str(x) for x in range(len(fold_pairs[0][0]))],
                                    columns= ['fold_' + str(x) for x in range(len(fold_pairs))])

    try:
        for x in range (len(fold_pairs)):
            fold_pairs_table['fold_' + str(x)] = np.transpose(np.array(fold_pairs[x][0]))
            fold_pairs_table = fold_pairs_table.replace(dict_feat)
            fold_pairs_table.to_csv(out_dir + '/Fold_pairsxtraining.csv')
    except:
         print('check fold pairs in python file')
    
    #In this part, we specify the cv for the validation of parameters
    y_counts = []
    for nfold in range(len(fold_pairs)):
        y = labels[fold_pairs[nfold][0]]
        unique_y, y_inversed = np.unique(y, return_inverse=True)
        y_counts.append(min(np.bincount(y_inversed)))
        
    parameters_cv = min(y_counts) #we choose the minimum cv samples.
    if parameters_cv>80:
        parameters_cv=5
    
    # Rank variables for each fold (does ttest unless otherwise specified)
    rank_per_fold = ps.get_rank_per_fold(data, labels, fold_pairs,
                                         save_path=out_dir,
                                         parallel=True)
    
    
    #Saving the ranking of features in a table
    dict_feat = dict(zip(range(len(list_features)), list_features))
    rankedfeatures = pd.DataFrame(index = ['rank_' + str(x) for x in range(data.shape[1])],
                                    columns= ['fold_' + str(x) for x in range(len(rank_per_fold))])

    rankedfeatures[rankedfeatures.columns] = np.transpose(np.array(rank_per_fold))
    rankedfeatures = rankedfeatures.replace(dict_feat)
    rankedfeatures.to_csv(out_dir + '/Ranked_features_per_fold.csv')
    
    # Force to train/test (two groups classification)
    if fix_train_test == True:
        del fold_pairs
        ksplit=1
        fold_pairs = [(list_train, list_test)]      
    
    if fix_features: #Is true?
        for ff in list_fix_features:
            for list_indx in range(len(rank_per_fold)):
                rank_per_fold[list_indx].remove(ff) #remove feature from list
                rank_per_fold[list_indx].insert(0,ff) #insert_position 0

    #--------------------------------
    #    COMPUTE SCORES
    #---------------------------------   

    dscore = []
    totalErrs = []
    predictions = pd.DataFrame()
    predictions2 = pd.DataFrame() 
    if compute_results:
        for name in classifiers_name:
            mdl = classifiers[name]
            param = params[name]
            # get_score runs the classifier on each fold,
            # each subset of selected top variables and does a grid search for
            # classifier-specific parameters (selects the best)
            clf, allConfMats, allTotalErrs,\
                allFittedClassifiers, predictions, predictions2 = \
                ps.get_score(data, labels, fold_pairs, name, mdl, param, parameters_cv,
                             numTopVars=topkfeatures,
                             rank_per_fold=rank_per_fold,
                             parallel=parallel_flag,
                             covariate_detrend_params=covariate_detrend_params,
                             rand_iter=-1)
            
            #Add label column to predictions dataframe and save
            predictions['Label'] = labels
            predictions2['Label'] = labels
                        
            predictions['Classified correctly %'] = predictions.apply(rightclassrate, axis=1)
            
            #changing index name
            dic1 = dict(zip(predictions.index.values.tolist(), subject_labels))
            
            predictions = predictions.rename(index=dic1)
            predictions2 = predictions2.rename(index=dic1)
            
            name = name.replace(" ", "_")
            predictions.to_csv(out_dir + '/Predictions4'+ name + '.csv')
            predictions2.to_csv(out_dir + '/Probabilities4'+ name + '.csv')
            
            # If you want to save classifier object, uncomment the next line
            ps.save_classifier_object(clf, allFittedClassifiers, name, out_dir)
            
            try:
                results = evaluation_metrics(predictions,predictions2, 'Label')
                results.to_csv(out_dir + '/Results4'+ name + '.csv') 
                print(out_dir)
            except:
                print('Incomplete predictions to generate results')
        
            # Append classifier results to list of all results
            dscore.append(allConfMats)
            totalErrs.append(allTotalErrs)
    #-----------------
    # DO SOME PLOTS
    #------------------
    
    logging.shutdown()
    plt.close('all')
    
    training_samples = len(fold_pairs[1][0])
    
    return(results, training_samples)


