#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:31:09 2017
This code run the Classification pipeline using as a main function classification.py
In addition, to compute the errors, this code generates csv files containing 
predictions, probabilities (if they are provided) and a performance summary results
table for each tested classifier.

Inputs
--------
A csv file containing a table of samples (1st columns are used as index) x
[features, target(class), subject_label]

Outputs
--------
A folder containing csv files with the results, model of the classifiers in pkl files
and a plot of the overall performance (error %) of all classifiers and topkfeatures

@author: Carla Agurto
"""

#Main code

import pandas as pd
from Classification import Classifiers
import numpy as np


def execute_pipeline(csvfilename, outdir, list_fix_features=[],
                     kfold_type='dependent', kfold_val=10, group_label=None,
                     target_label=None, covariate_detrend_params=None):
    #------------------------------------------
    # Information of the data to be processed
    #------------------------------------------
    #1) Results stored in outdir
         
    #2)specify the files to be processed.
    files_list = [csvfilename]
    
    # use line below instead if you want to process many csv files in a specific directory
    #tmp = os.listdir(dirfile)
    ##only use csv files
    #files_list = []
    #for t in tmp:
    #    if t.endswith('.csv'):
    #        files_list.append(t)
    
    
    # 3)Specify the columns name for the target(class) or subject_labels
    #Note: remaining columns are used as features
    subjects = group_label
    target = target_label

    # 5)Specify the kfold for the classification. 1 for leaveone out
    # kfold_type = 'dependent'     # fixed or dependent
    # kfold_val = 10
    
    #4)Specify how the top ranked features are going to be selected
    #Select the number of features to be chosen in logarithmic way
    maxtopk_values = 5 #number of topK features to try
    
    #If you want to use all features in addition to topkfeatures list set variable use_allfeat to True.
    use_allfeat = True
    
    # 6)Specify a list of features and a boolean variable if you want to fix some 
    #features for the ranking meaning tohave high priority
    if not list_fix_features:
        fix_features = False
    else:
        fix_features = True
    
    #6) Specify a list for train and test from your table. to train using some features 
    #In order to have results for all the samples. The responses of the training features
    #are obtained by training the model with the testing features.
    fix_train_test = False
    #example below
    list_aux = range(62)
    list_aux.remove(0)
    list_aux.remove(30)
    list_aux.remove(31)
    list_aux.remove(60)
    list_train = np.array(list_aux)
    list_test = np.array([0,30, 31, 60])  
    
    #-----------------------------
    # Manually set of parameters
    #----------------------------
    #Specify in a list the classifiers you want to use from here:
    classifiers_name = ["Logistic Regression"]#, "Random Forest" ,"Decision Tree","Nearest Neighbors","Logistic Regression",
                        #"Naive Bayes", "LDA", "SDCA_SVM_elastic","CGC_log_L1", "Chance"] #,
    #classifiers_name = ["Linear SVM", "Decision Tree","Nearest Neighbors"]
    #FOR REGRESION USE THE FOLLOWING CLASSIFIERS
    #"Chance","Elastic_Net","SDCA_Elastic","Ridge_CD","Lasso_CD","Lasso",
    #"Ridge","eSVR", "NuSVR","lightSVR","RANDSAC"#
    #classifiers_name = ["Ridge_CD","Lasso_CD","Lasso",
    #                    "Ridge","eSVR", "NuSVR","lightSVR","RANDSAC"]
    #isregression == 1
    
    #------------------------------------------------------------------------------
    # Main loop
    #------------------------------------------------------------------------------  
    
    for filename in files_list:
        try:
            print(' Processing : ' , filename)
            
            # By convention, the panda table is expected to contain one column with 
            # class (target) one column with the subject label (integer values) 
            # and the rest of the columns are considered features.
            table = pd.read_csv(filename, index_col=0)

            if kfold_type == 'fixed':
                kfold = kfold_val
            else:
                kfold = len(table[subjects].unique())
                 
            list_features = table.columns.tolist()
            list_features.remove(target) #this columns contains the class
            list_features.remove(subjects) #this columns contains the subject label
            
           #creating a list with a topkfeatures to be analized.
            exp_number = np.log(len(list_features))/np.log(10)
            aux = np.array(np.logspace(0,exp_number,num=maxtopk_values),dtype=int)
            topkfeatures = list(set(list(aux))) 
            topkfeatures.sort()
            topkfeatures = np.unique(topkfeatures).tolist()
            print(topkfeatures)
            
            #add the last feature if it was requested
            if use_allfeat and (len(topkfeatures) > 1):
                if len(list_features) in topkfeatures:
                    print('all features included in the topkfeatures')
                else:
                    topkfeatures.append(len(list_features))
            # Obtaining the number of samples per subject.   
            runs_per_subj = len(table[subjects])/float(len(table[subjects].unique()))
           # topkfeatures=[1,2]

            results, training_samples = Classifiers(outdir,topkfeatures,classifiers_name, 
                            runs_per_subj, table, target,
                            list_features, subjects,
                            fix_features,list_fix_features,
                            fix_train_test, list_train, list_test,
                            kfold, covariate_detrend_params) 
            print(results)
    
        except Exception as e: 
            print(str(e))
            print('File is not a csv file or it contains error in the data')
