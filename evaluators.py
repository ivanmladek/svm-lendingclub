from sklearn import (svm, preprocessing,
                     grid_search, metrics,
                     cross_validation)
from sklearn import ensemble as ens
from sklearn import linear_model as lm
from sklearn import tree as tr
from sklearn import datasets
from random import sample
from sklearn import grid_search
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import (zero_one_loss,
                             recall_score,
                             roc_curve, auc)
from scipy import interp
import pylab as pl
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from scipy.stats import ks_2samp

"""
Copyright 2013 by SVM Risk Consulting
All rights reserved. No part of this document may be reproduced or transmitted in any form or by any means, electronic, mechanical, photocopying, recording, or otherwise, without prior written permission of SVM Risk Consulting.
"""


#From
#http://stackoverflow.com/questions/14363833/joblib-parallel-pool-closed-error
n_samp = 1000
svm_cs = 100000
svm_maxiter=100000
en_params = {
    #'Poly SVM':{
    #    'eng':svm.SVC(cache_size=svm_cs,max_iter=svm_maxiter),
    #    'params':{'kernel':('poly','rbf'),'degree':[2,5],
    #              'C':[.1,1,5,10,20,40,80,160,320]}},
    'RandomForestClassifier':{
        'eng':ens.RandomForestClassifier(),
        'params':{'n_estimators':[1000,2000,3000],
                  'min_samples_split':[8,10,20]}},
    #'ExtraForestClassifier':{
    #    'eng':ens.ExtraTreesClassifier(),
    #    'params':{'n_estimators':[100,500,1000,2000,3000],
    #              'min_samples_split':[2,4,6,8,10],
    #              'min_samples_leaf':[2,4,6,8,10]}},
    #'LogisticRegression':{'eng':lm.LogisticRegression(),
    #                      'params':{'C':[.01,.1,10,20,30,40,50,100,200],
    #                                }},

    #'DecisionTrees':{'eng':tr.DecisionTreeClassifier(),
    #                 'params':{
    #        'min_samples_split':[2,4,6,8,10],
    #        'min_samples_leaf':[2,4,6,8,10],
    #        'min_density':[.05,.1,.2,.4,.6],
    #        }},
    }

parameters = [{'C': [1],
               #0.001, 0.01, 0.1, 1],#, 10],#, 100],#, 1000],
               'gamma': [0.1],#, 0.01,  0.001, 0.0001],
               'kernel': ['poly','linear', 'rbf'], 'degree': [2],
               'class_weight': [{0: 1, 1: 1},
                                #{0: 1, 1: 2},
                                #{0: 1, 1: 3},
                                #{0: 1, 1: 5},
                                ],
               }]

def rfe_optim(X_scaled, status):
    """
    Perform recursive feature elimination
    http://scikit-learn.org/dev/auto_examples/plot_rfe_with_\
    cross_validation.html#example-plot-rfe-with-cross-validation-py
    """
    # Create the RFE object and compute a cross-validated score.
    svc = svm.SVC(kernel="linear")
    rfecv = RFECV(estimator=svc, step=1,
                  cv=StratifiedKFold(status, 2),
                  loss_func=zero_one_loss)
    rfecv.fit(X_scaled, status)
    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    import pylab as pl
    pl.figure()
    pl.xlabel("Number of features selected")
    pl.ylabel("Cross validation score (nb of misclassifications)")
    pl.plot(range(1, len(rfecv.cv_scores_) + 1), rfecv.cv_scores_)
    pl.savefig('rfe_optim.png')
    return rfecv.n_features_

def ks_score(tpr, fpr):
    for i,(t, f) in enumerate(zip(tpr, fpr)):
        print i, t, f, float(t-i/100.), float(f-i/100.)

def roc(X, y,classifier,  n_f=10):
    """
    From http://scikit-learn.org/dev/auto_examples/plot_roc_crossval.html#example-plot-roc-crossval-py
    """
    # Run classifier with crossvalidation and plot ROC curves
    cv = StratifiedKFold(y, n_folds=n_f)
    #classifier = svm.SVC(kernel='linear', probability=True)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    pl.figure()
    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        #pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)'
        #        % (i, roc_auc))
    pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print ks_score(mean_fpr, mean_tpr)
    ks2 = ks_2samp(mean_fpr, mean_tpr)
    print ks2
    pl.plot(mean_fpr, mean_tpr, 'k--',
            label='LendingScore', lw=2)

    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('LendingSCORE Gains Chart')
    pl.legend(loc="lower right")
    pl.savefig('roc.png')


def forest_optim(X_scaled, status, FEAT_TOL=0.05):
    """
    http://scikit-learn.org/0.11/auto_examples/ensemble/\
    plot_forest_importances.html#example-ensemble-plot-forest-importances-py
    """
    print 'starting Random Forest Optimization'
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  compute_importances=True,
                                  random_state=0)
    forest.fit(X_scaled, status)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print "Feature ranking:"
    features_to_train = list()

    for f in xrange(len(X_scaled[0])):
        if importances[indices[f]] > FEAT_TOL:
            features_to_train.append(indices[f])
            print "%d. feature %d (%f)" % (
                f + 1,
                indices[f], importances[indices[f]])

    # Plot the feature importances of the trees and of the forest

    pl.figure()
    pl.title("Feature importances")

    for tree in forest.estimators_:
        pl.plot(xrange(len( tree.feature_importances_[indices])),
                tree.feature_importances_[indices], "r")

    pl.plot(xrange(len(importances[indices])),
            importances[indices], "b")
    pl.savefig('random_forest_optim.png')
    return features_to_train
