"""
Module of multiple classifiers (polyssifier) that enables univariate feature
selection (the latter capability was not used for this work).

@authors: Sergey Plis, Devon Hjelm, Pouya Bashivan, Pablo Polosecki, Carla Agurto 
"""


USEJOBLIB=False

import argparse
import logging

if USEJOBLIB:
    from joblib.pool import MemmapingPool as Pool
else:
    from multiprocessing import Pool

import matplotlib as mpl
mpl.use("Agg")
import multiprocessing
import numpy as np
from os import path
import pandas as pd
import pickle
import scipy.io
from scipy.spatial.distance import pdist
from scipy.stats import ttest_ind

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from lightning.classification import CDClassifier, SDCAClassifier
import copy



# Font sizes for plotting
font = {'family' : 'normal',
        'size'   : '16'}
mpl.rc('font', **font)
FONT_SIZE = 18
PROCESSORS = 1 # 12 # 31 #12 # 31 #
np.random.seed(6606)
NAMES = ["Chance", "Nearest Neighbors", "Linear SVM", "RBF SVM",  "Decision Tree",
         "Random Forest", "Logistic Regression", "Naive Bayes", "LDA"]



def rank_vars(xTrain, yTrain, scoreFunc, covariate_detrend_params=None):
    """
    ranks variables using the supplied scoreFunc.
    Inputs:
        xTrain: training features matrix
        yTrain: training labels vector
        scoreFunc: function used to rank features (pearsonr or mutual_info_score)
        covariate_detrend_params: dict or None. If dict apply detrending using
        those parameters
    Output:
        returns the ranked list of features indices
    """

    usedX = xTrain

    funcsDic = {
        'pearsonr': [np.arange(usedX.shape[1]), 1],
        'mutual_info_score': np.arange(usedX.shape[0]),
        'ttest_ind': [np.arange(usedX.shape[1]), 1],
        }

    scores = list()
    for feat in np.arange(usedX.shape[1]):
        if scoreFunc.func_name == 'pearsonr':
            scores.append(scoreFunc(usedX[:, feat], yTrain))
        elif scoreFunc.func_name == 'ttest_ind':
            scores.append(scoreFunc(usedX[yTrain == 1, feat], usedX[yTrain==-1, feat]))

    scores = np.asarray(scores)
    pvals = scores[funcsDic[scoreFunc.func_name]]
    dtype = [('index', int), ('p-val',float)]
    ind_and_pvals = [x for x in enumerate(pvals)]
    sortedIndices = [i[0] for i in np.sort(np.array(ind_and_pvals,dtype=dtype),order='p-val')]
    # sortedIndices = [i[0] for i in sorted(enumerate(pvals), key=lambda x:x[1])]
    return sortedIndices


class SupervisedStdScaler(StandardScaler):
    '''
    A standard scaler that uses group labels to Scale
    '''

    def __init__(self):
        self.__subscaler = StandardScaler()
        self.__subscaleru = StandardScaler()

    def fit(self, X, y=None, label=None):
        if not (y is None or label is None):
            x_used = X[y == label]
        else:
            x_used = X
        self.__subscaler.fit(x_used)
        self.__subscaleru.fit(X)

    def transform(self, X, y=None, label=None):
        scale = self.__subscaler.scale_
        center = self.__subscaleru.mean_
        tile_center = np.tile(center,[X.shape[0], 1])
        tile_scale = np.tile(scale,[X.shape[0], 1])
        X_transf = (X-tile_center) / tile_scale
        return X_transf


class SupervisedRobustScaler(StandardScaler):
    '''
    A standard scaler that uses group labels to Scale
    '''

    def __init__(self):
        self.__subscaler = RobustScaler()
        self.__subscaleru = RobustScaler()

    def fit(self, X, y=None, label=None):
        if not (y is None or label is None):
            x_used = X[y == label]
        else:
            x_used = X
        self.__subscaler.fit(x_used)
        self.__subscaleru.fit(X)

    def transform(self, X, y=None, label=None):
        scale = self.__subscaler.scale_
        center = self.__subscaleru.center_
        tile_center = np.tile(center,[X.shape[0], 1])
        tile_scale = np.tile(scale,[X.shape[0], 1])
        X_transf = (4./3) * (X-tile_center) / tile_scale
        return X_transf
    #If data is normally distributed, it will have unit variance


class Ranker(object):
    """
    Class version of univariate ranking, to pass to multicore jobs
    Inputs:
        data: the full data matrix
        labels: full class labels
        ranking function: the ranking function to give to rank_vars
        rank_vars: the rank_vars function
        fp: list of fold train-test pairs
        covariate_detrend_params: dict or None. default: None
            If dict, detrend is applied with these parameters
    """
    def __init__(self, data, labels, ranking_function, rank_vars=rank_vars,
                 covariate_detrend_params=None, give_idx_to_ranker=False):
        self.data = data
        self.labels = labels
        self.rf = ranking_function
        self.rank_vars = rank_vars
        self.covariate_detrend_params = covariate_detrend_params
        if covariate_detrend_params:
            give_idx_to_ranker = True
        self.give_idx_to_ranker = give_idx_to_ranker

    def __call__(self, fp):
        if self.give_idx_to_ranker:
            rv = self.rank_vars(fp, self.labels[fp[0]],
                                self.rf,
                                covariate_detrend_params=self.covariate_detrend_params)
        else:
            rv = self.rank_vars(self.data[fp[0], :], self.labels[fp[0]],
                                self.rf,
                                covariate_detrend_params=self.covariate_detrend_params)
        return rv


def get_rank_per_fold(data, labels, fold_pairs, ranking_function=ttest_ind,
                      save_path=None,load_file=False,
                      parallel=True, covariate_detrend_params=None):

    '''
    Applies rank_vars to each test set in list of fold pairs
    Inputs:
        data: array
            features for all samples
        labels: array
            label vector of each sample
        fold_pair: list
            list pairs of index arrays containing train and test sets
        ranking_function: function object, default: ttest_ind
            function to apply for ranking features
        ranking_function: function
            ranking function to use, default: ttest_ind
        save_path: dir to load and save ranking files
        load_file: bool
            Whether to try to load an existing file, default: True
        parallel: bool
            True if multicore processing is desired, default: True
        covariate_detrend_params: Dict or None, default: None
            If Dict, apply covariate detreding with these dict of parameters
    Outputs:
        rank_per_fod: list
            List of ranked feature indexes for each fold pair
    '''
    file_loaded = False
    if load_file:
        if isinstance(save_path, str):
            fname = path.join(save_path, "{}_{}_folds.mat".format(
                              ranking_function.__name__, len(fold_pairs)))
            try:
                rd = scipy.io.loadmat(fname, mat_dtype = True)
                rank_per_fold = rd['rank_per_fold'].tolist()
                file_loaded = True
            except:
                pass
        else:
            print('No rank file path: Computing from scratch without saving')
    if not file_loaded:
        if not parallel:
            rank_per_fold = []
            for fold_pair in fold_pairs:
                if covariate_detrend_params:
                    rankedVars = rank_vars(fold_pair[0],
                                           labels[fold_pair[0]],
                                           ranking_function,
                                           covariate_detrend_params)
                else:
                    rankedVars = rank_vars(data[fold_pair, :],
                                           labels[fold_pair[0]],
                                           ranking_function)
                    rank_per_fold.append(rankedVars)
        else:
            pool = Pool(processes=min(len(fold_pairs), PROCESSORS))
            rank_per_fold = pool.map(Ranker(data, labels, ranking_function,
                                            rank_vars,
                                            covariate_detrend_params),
                                    fold_pairs)
            pool.close()
            pool.join()
        if isinstance(save_path, str):
            fname = path.join(save_path, "{}_{}_folds.mat".format(
                              ranking_function.__name__, len(fold_pairs)))

    return rank_per_fold


def make_classifiers(NAMES):
    """Function that makes classifiers each with a number of folds.

    Returns two dictionaries for the classifiers and their parameters, using
    `data_shape` and `ksplit` in construction of classifiers.

    Parameters
    ----------
    data_shape : tuple of int
        Shape of the data.  Must be a pair of integers.
    ksplit : int
        Number of folds.

    Returns
    -------
    classifiers: dict
        The dictionary of classifiers to be used.
    params: dict
        A dictionary of list of dictionaries of the corresponding
        params for each classifier.
    """


    classifiers = {
        "Chance": DummyClassifier(strategy="most_frequent"),
        "Nearest Neighbors": KNeighborsClassifier(3),
        "Linear SVM": LinearSVC(penalty='l2', C=1,# probability=True,
                          class_weight='balanced'),
        "RBF SVM": SVC(gamma=2, C=1, probability=True),
        "Decision Tree": DecisionTreeClassifier(max_depth=None,
                                                max_features="auto"),
        "Random Forest": RandomForestClassifier(max_depth=None,
                                                n_estimators=20,
                                                max_features="auto",
                                                n_jobs=PROCESSORS),
        "Logistic Regression": LogisticRegression(penalty='l2',
                                                   class_weight='balanced'),
        "Naive Bayes": GaussianNB(),
        "LDA": LDA(),
        "SGD_logL1": SGDClassifier(random_state=1952,loss='log', average = 3,
                                  penalty='l1',
                                  alpha=1e-3,
                                  class_weight='balanced'),
        "SGD_log_elastic": SGDClassifier(random_state=1952,loss='log',
                                          class_weight='balanced',
                                          alpha=1e-3,
                                          average = 3,
                                          penalty='elasticnet'),
        "SGD_SVM_elastic": SGDClassifier(random_state=1952,loss='hinge',
                                          class_weight='balanced',
                                          average = 3,
                                          alpha=1e-3,
                                          penalty='elasticnet'),

        "CGC_log_L1": CDClassifier(penalty="l1",
                   loss="log",
                   multiclass=False,
                   max_iter=200,
                   C=1,
                   tol=1e-3),
        "SDCA_SVM_elastic": SDCAClassifier(
                   loss="hinge",
                   max_iter=200,
                   tol=1e-3)

        }

    params = {
        "Chance": {},
        "Nearest Neighbors": {"n_neighbors": [1,3, 5]},#, 10]},
        "Linear SVM": {"C": np.logspace(-5, 2, 8),
                       "loss":['hinge']},# 'squared_hinge']},
        "RBF SVM": {"kernel": ["rbf"],
                     "gamma": np.logspace(-2, 0, 6).tolist() + \
                              np.logspace(0,1,5)[1:].tolist(),
                     "C": np.logspace(-2, 2, 5).tolist()},
        "Decision Tree": {},
        "Random Forest": {"max_depth": np.round(np.logspace(np.log10(2), \
                                       1.2, 6)).astype(int).tolist()},
        "Logistic Regression": {"C": np.logspace(-2, 3, 6).tolist()},
        "Naive Bayes": {},
        "LDA": {},
        "SGD_logL1": {"alpha": np.logspace(-5, 2, 7)},
        "SGD_log_elastic": {"alpha": np.logspace(-5, 2, 6),
                            "l1_ratio": 10**np.array([-2, -1, -.5, -.25,
                                                      -.12, -.06, -.01])},
        "SGD_SVM_elastic": {"alpha": np.logspace(-5, 2, 6),
                            "l1_ratio": 10**np.array([-2, -1, -.5, -.25,
                                                      -.12, -.06, -.01])},
        "CGC_log_L1": {"alpha": np.logspace(-5, 2, 6)},
        "SDCA_SVM_elastic": {"alpha": np.logspace(-4, 4, 5),
                             "l1_ratio": 10**np.array([-3,-2, -1, np.log10(.5),
                                                       np.log10(.9)])}
            }

    return classifiers, params


class per_split_classifier(object):
    """
    Class version of classify function, to pass to multicore jobs
    Inputs:
        data: the full data matrix
        labels: full class labels
        classifier: classifier object to use
        numTopVars: list of top variables to use
        zipped_ranks_n_fp: zipped list 2-tuple with ranked vars and train-test
                           indices
        fp: a single train-test pair
    """
    def __init__(self, data, labels, classifier, numTopVars,
                 covariate_detrend_params=None, longitudinal_pca_params=None):
        self.data = data
        self.labels = labels
        self.clf = classifier
        self.numTopVars = numTopVars
        self.covariate_detrend_params = covariate_detrend_params
        self.longitudinal_pca_params = longitudinal_pca_params

    def __call__(self, zipped_ranks_n_fp):
        rankedVars, fp = zipped_ranks_n_fp
        confMats = []
        totalErrs = []
        fitted_classifiers = []
        predictions = []
        predictions2 = []
        for numVars in self.numTopVars:
            if self.covariate_detrend_params:
               self.covariate_detrend_params['rankedVars'] =  rankedVars
               self.covariate_detrend_params['numVars'] = numVars
            classify_output = classify(self.data[:, rankedVars[:numVars]],
                                       self.labels, fp, self.clf,
                                       covariate_detrend_params=self.covariate_detrend_params,
                                       longitudinal_pca_params=self.longitudinal_pca_params)
            confMats.append(classify_output[0])
            totalErrs.append(classify_output[1])
            fitted_classifiers.append(classify_output[2])
            predictions.append(classify_output[3])
            predictions2.append(classify_output[4])
        
        return confMats, totalErrs, fitted_classifiers, predictions, predictions2


def classify(data, labels, train_test_idx, classifier=None,
             covariate_detrend_params=None, longitudinal_pca_params=None):

    """
    Classifies given a fold and a model.

    Parameters
    ----------
    data: array_like
        2d matrix of observations vs variables
    labels: list or array_like
        1d vector of labels for each data observation
    (train_idx, test_idx) : list
        set of indices for splitting data into train and test
    classifier: sklearn classifier object
        initialized classifier with "fit" and "predict_proba" methods.
    Returns
    -------
    WRITEME
    """
   
    assert classifier is not None, "Why would you pass not classifier?"
    train_idx, test_idx = train_test_idx
    clean_data = data

    # Data scaling based on training set
    scaler = SupervisedStdScaler() #SupervisedRobustScaler()  #  #
    scaler.fit(clean_data[train_idx,:], labels[train_idx], label=-1)
    scaler.fit(clean_data[train_idx, :], labels[train_idx])
    data_train = scaler.transform(clean_data[train_idx, :])
    data_test = scaler.transform(clean_data[test_idx, :])
    try:
        classifier.fit(data_train, labels[train_idx])

        predictions = classifier.predict(data_test)
        
        try:
            #mod by Carla aAgurto
            predictions2=[]
            probabilities = classifier.predict_proba(data_test)
            for j in range(len(probabilities)):
                predictions2.append(1-probabilities[j][0])            
                
        except:
            predictions2 = predictions
            #print('WARNING : no attribute predict_proba for ROC curve')    
      
        confMat = confusion_matrix(labels[test_idx],
                                   predictions)
        #print('confmat: ', confMat)

        if confMat.shape == (1, 1):
            if all(labels[test_idx] == -1):
                confMat = np.array([[confMat[0], 0], [0, 0]],
                                   dtype=confMat.dtype)
            else:
                confMat = np.array([[0, 0], [0, confMat[0]]],
                                   dtype=confMat.dtype)
        confMatRate = confMat / np.tile(np.sum(confMat, axis=1).
                                        astype('float'), (2, 1)).transpose()
        totalErr = (confMat[0, 1] + confMat[1, 0]) / float(confMat.sum())

        if hasattr(classifier, 'param_grid'):
                fitted_model = classifier.best_estimator_
        else:
                fitted_model = copy.copy(classifier)
        return confMatRate, totalErr, fitted_model, predictions, predictions2
    except np.linalg.linalg.LinAlgError:
        return np.array([[np.nan, np.nan], [np.nan, np.nan]]), np.nan, None, np.nan


def get_score(data, labels, fold_pairs, name, model, param, parameters_cv, numTopVars,
              rank_per_fold=None, parallel=False, rand_iter=-1,
              covariate_detrend_params=None,
              longitudinal_pca_params=None):
    """
    Function to get score for a classifier.

    Parameters
    ----------
    data: array_like
        Data from which to derive score.
    labels: array_like or list
        Corresponding labels for each sample.
    fold_pairs: list of pairs of array_like
        A list of train/test indicies for each fold
        dhjelm(Why can't we just use the KFold object?)
    name: str
        Name of classifier.
    model: WRITEME
    param: WRITEME
        Parameters for the classifier.
    parallel: bool
        Whether to run folds in parallel. Default: True

    Returns
    -------
    classifier: WRITEME
    allConfMats: Confusion matrix for all folds and all variables sets and best performing parameter set
                 ([numFolds, numVarSets])
    """
    
    assert isinstance(name, str)
    #logging.info("Classifying %s" % name)
    ksplit = len(fold_pairs)

    # Redefine the parameters to be used for RBF SVM (dependent on
    # training data)
    if "SGD" in name:
        param["n_iter"] = [25]  # [np.ceil(10**3 / len(fold_pairs[0][0]))]
    classifier = get_classifier(name, model, param, parameters_cv, rand_iter=rand_iter)

    if name == "RBF SVM": #This doesn't use labels, but looks as ALL data
        logging.info("RBF SVM requires some preprocessing."
                     "This may take a while")
        #Euclidean distances between samples
        dist = pdist(StandardScaler().fit(data), "euclidean").ravel()
        #dist = pdist(RobustScaler().fit_transform(data), "euclidean").ravel()
        #Estimates for sigma (10th, 50th and 90th percentile)
        sigest = np.asarray(np.percentile(dist,[10,50,90]))
        #Estimates for gamma (= -1/(2*sigma^2))
        gamma = 1./(2*sigest**2)
        #Set SVM parameters with these values
        param = [{"kernel": ["rbf"],
                  "gamma": gamma.tolist(),
                  "C": np.logspace(-2,2,5).tolist()}]

    if (not parallel) or \
    (name == "Random Forest") or ("SGD" in name):
       # logging.info("Attempting to use grid search...")
        classifier.n_jobs = PROCESSORS
        # classifier.pre_dispatch = 1 # np.floor(PROCESSORS/24)
        allConfMats = []
        allTotalErrs = []
        allFittedClassifiers = []
        colnames = ['top_' + str(x) for x in numTopVars]
        allPredictions = pd.DataFrame(index = range(len(labels)), columns = colnames )
        allPredictions2 = pd.DataFrame(index = range(len(labels)), columns = colnames )
        for i, fold_pair in enumerate(fold_pairs):
            confMats = []
            totalErrs = []
            fitted_classifiers = []

           # logging.info("Classifying a %s the %d-th out of %d folds..."
            #       % (name, i+1, len(fold_pairs)))
            if rank_per_fold is not None:
                rankedVars = np.squeeze(rank_per_fold)[i]
            else:
                rankedVars = np.arange(data.shape[1])
            for numVars in numTopVars:
                #logging.info('Classifying for top %i variables' % numVars)
                if covariate_detrend_params:
                   covariate_detrend_params['rankedVars'] =  rankedVars
                   covariate_detrend_params['numVars'] = numVars
                   
                   print(rankedVars, numVars)
                   
                classify_output = classify(data[:, rankedVars[:numVars]],
                                           labels,
                                           fold_pair,
                                           classifier,
                                           covariate_detrend_params=covariate_detrend_params,
                                           longitudinal_pca_params=longitudinal_pca_params)
                confMat, totalErr, fitted_classifier, prediction, prediction2\
                    = classify_output
               # print(labels,fold_pair,classifier,covariate_detrend_params,longitudinal_pca_params, rankedVars[:numVars])
                    
                confMats.append(confMat)
                totalErrs.append(totalErr)
                fitted_classifiers.append(fitted_classifier)
                testedsamples = fold_pair[1]
                #print ('check this:' , prediction, fold_pair[1])
                #save predictions in table
                for j in range(len(prediction)):
                    allPredictions.iloc[testedsamples[j]]['top_' + str(numVars)] = prediction[j]
                    allPredictions2.iloc[testedsamples[j]]['top_' + str(numVars)] = prediction2[j]
                
            # recheck the structure of area and fScore variables
            allConfMats.append(confMats)
            allTotalErrs.append(totalErrs)
            allFittedClassifiers.append(fitted_classifiers)

    else:
        classifier.n_jobs = PROCESSORS
        logging.info("Multiprocessing folds for classifier {}.".format(name))
        pool = Pool(processes=min(ksplit, PROCESSORS))
        out_list = pool.map(per_split_classifier(data, labels, classifier,
                                                 numTopVars,
                                                 covariate_detrend_params=covariate_detrend_params,
                                                 longitudinal_pca_params=longitudinal_pca_params),
                            zip(rank_per_fold, fold_pairs))
        pool.close()
        pool.join()

        allConfMats, allTotalErrs, allFittedClassifiers, allPredictions, allPredictions2\
            = tuple(zip(*out_list))
    return classifier, allConfMats, allTotalErrs, allFittedClassifiers, allPredictions, allPredictions2


def get_classifier(name, model, param, parameters_cv, rand_iter=-1):
    """
    Returns the classifier for the model.

    Parameters
    ----------
    name: str
        Classifier name.
    model: WRITEME
    param: WRITEME
    data: array_like, optional

    Returns
    -------
    WRITEME
    """
    
    assert isinstance(name, str)
    if param: # Do grid search only if parameter list is not empty
        N_p = np.prod([len(l) for l in param.values()])
        if (N_p <= rand_iter) or rand_iter<=0:
           # logging.info("Using grid search for %s" % name)
            model = GridSearchCV(model, param, cv=parameters_cv, scoring="accuracy",
                                 n_jobs=PROCESSORS)
           
        else:
            #logging.info("Using random search for %s" % name)
            model = RandomizedSearchCV(model, param, cv=parameters_cv, scoring="accuracy",
                                 n_jobs=PROCESSORS, n_iter=rand_iter)
    return model


def save_classifier_results(classifier_name, out_dir, allConfMats,
                            allTotalErrs):
    """
    saves the classifier results including TN, FN and total error.
    """

    # convert confusion matrix and total errors into numpy array
    tmpAllConfMats = np.array(allConfMats)
    tmpAllTotalErrs = np.array(allTotalErrs)
    # initialize mean and std variables
    TN_means = np.zeros(tmpAllConfMats.shape[1])
    TN_stds = np.zeros(tmpAllConfMats.shape[1])
    FN_means = np.zeros(tmpAllConfMats.shape[1])
    FN_stds = np.zeros(tmpAllConfMats.shape[1])
    total_means = np.zeros(tmpAllConfMats.shape[1])
    total_stds = np.zeros(tmpAllConfMats.shape[1])

    for j in range(tmpAllConfMats.shape[1]):
        tmpData = tmpAllConfMats[:, j, 0, 0]
        TN_means[j] = np.mean(tmpData[np.invert(np.isnan(tmpData))])
        TN_stds[j] = np.std(tmpData[np.invert(np.isnan(tmpData))])
        tmpData = tmpAllConfMats[:, j, 1, 0]
        FN_means[j] = np.mean(tmpData[np.invert(np.isnan(tmpData))])
        FN_stds[j] = np.std(tmpData[np.invert(np.isnan(tmpData))])
        tmpData = tmpAllTotalErrs[:, j]
        # Compute mean of std of non-Nan values
        total_means[j] = np.mean(tmpData[np.invert(np.isnan(tmpData))])
        total_stds[j] = np.std(tmpData[np.invert(np.isnan(tmpData))])

    with open(path.join(out_dir, classifier_name + '_errors.mat'), 'wb') as f:
        scipy.io.savemat(f, {'TN_means': TN_means,
                             'TN_stds': TN_stds,
                             'FN_means': FN_means,
                             'FN_stds': FN_stds,
                             'total_means': total_means,
                             'total_stds': total_stds,
                             })


def save_classifier_predictions_per_sample(classifier_name, out_dir,
                                           predictions,predictions2, fold_pairs, labels,
                                           subjects_per_run, ktop=-1):
    '''
    Construction of a data frame with labels, calssifier predictions and
    subject name of each sample
    modifie by carla to save predictions
    '''
    test_idx = (fp[1] for fp in fold_pairs)
    idx_list = []
    pred_list = []
    in_labels = []
    sample_subj = []
    for indx_from_fold, pred_from_fold in zip(test_idx, predictions2):
        idx_list.extend(indx_from_fold.tolist())
        pred_list.extend(pred_from_fold[ktop].astype(int).tolist())
        in_labels.extend(labels[indx_from_fold].astype(int).tolist())
        sample_subj.extend(subjects_per_run[indx_from_fold].tolist())
    label_pred_per_sample = pd.DataFrame(dict(zip(('sample', 'subjid',
                                                   'labels', 'prediction2'),
                                                  (idx_list, sample_subj,
                                                   in_labels, pred_list)))
                                         ).set_index('sample')
    label_pred_per_sample.to_csv(path.join(out_dir, classifier_name +
                                           '_predictions2.csv'))


def save_classifier_object(clf, FittedClassifiers, name, out_dir):
    if out_dir is not None:
        save_path = path.join(out_dir, name + '.pkl')
        #logging.info("Saving classifier to %s" % save_path)
        classifier_dict = {'name': name,
                           'classifier': clf,
                           'FittedClassifiers': FittedClassifiers}
        with open(save_path, "wb") as f:
            pickle.dump(classifier_dict, f)


def save_combined_results(NAMES, dscore, totalErrs, numTopVars, out_dir, filebase):
    confMatResults = {name.replace(" ", ""): scores for name, scores in zip(NAMES, dscore)}
    confMatResults['topVarNumbers'] = numTopVars
    totalErrResults = {name.replace(" ", ""): errs for name, errs in zip(NAMES, totalErrs)}
    totalErrResults['topVarNumbers'] = numTopVars
    # save results from all folds
    # dscore is a matrix [classifiers, folds, #vars, 2, 2]
    dscore = np.asarray(dscore)
    totalErrs = np.asarray(totalErrs)
    with open(path.join(out_dir, filebase + '_dscore_array.mat'), 'wb') as f:
        scipy.io.savemat(f, {'dscore': dscore,
                             'topVarNumbers': numTopVars,
                             'classifierNames': NAMES})

    with open(path.join(out_dir, filebase + '_errors_array.mat'), 'wb') as f:
        scipy.io.savemat(f, {'errors': totalErrs,
                             'topVarNumbers': numTopVars,
                             'classifierNames': NAMES})
    # Save all results
    with open(path.join(out_dir, 'confMats.mat'),'wb') as f:
        scipy.io.savemat(f, confMatResults)
    with open(path.join(out_dir, 'totalErrs.mat'),'wb') as f:
        scipy.io.savemat(f, totalErrResults)


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory",
                        help="Directory where the data files live.")
    parser.add_argument("out", help="Output directory of files.")
    parser.add_argument("-t", "--test", action="store_true",
                        help=("Test mode, avoids slow classifiers and uses"
                              " 3 folds"))
    parser.add_argument("--folds", default=10,
                        help="Number of folds for n-fold cross validation")
    parser.add_argument("--data_pattern", default="*.mat",
                        help="Pattern for data files")
    parser.add_argument("--label_pattern", default="*.mat",
                        help="Pattern for label files")
    return parser


if __name__ == "__main__":
    CPUS = multiprocessing.cpu_count()
    if CPUS < PROCESSORS:
        raise ValueError("Number of PROCESSORS exceed available CPUs, "
                         "please edit this in the script and come again!")

    numTopVars = [50, 100, 300, 900, 2700]

    parser = make_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s",
                        filename=path.join(args.out, 'log.log'),
                        filemode='w',
                        level=logging.DEBUG)
    logger = logging.getLogger()
    ch = logging.StreamHandler(logging.sys.stdout)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

