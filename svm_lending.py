#!/export/disk0/wb/python2.6/bin/python
from sklearn import (svm, preprocessing,
                     grid_search, metrics)
import pandas as pd

from datetime import datetime
import numpy as np

def binary_status(st):
    if (st == 'Issued' or
        st == 'Fully Paid' or
        st == 'Current' or
        st == 'Late (16-30 days)' or
        st == 'Does not meet the current credit policy.  Status:In Grace Period'):
        return -1
    else:
        return 1

def prepare_data_for_year(training, target_y, def_scaler=None):
    #Weed out NaNs
    finite_ix = np.flatnonzero(np.isfinite([f for f in training.fico_range_high]))
    finite = training.take(finite_ix)

    #Prepare training and test data
    status = np.array([binary_status(l) for l,d in zip(finite.loan_status,
                                                     finite.issue_d)
                       if (datetime.strptime(d,'%Y-%m-%d').year == target_y)])
    training_data = np.array([[f, ann_inc,
                               amount, dti, open_acc, num_inq] for f, ann_inc,amount,dti,
                              open_acc,num_inq,
                              issue_d
                              in zip(finite.fico_range_high,
                                     finite.annual_inc,
                                     finite.loan_amnt,
                                     finite.dti,
                                     finite.open_acc,
                                     finite.inq_last_6mths,
                                     finite.issue_d)
                              if (datetime.strptime(issue_d,'%Y-%m-%d').year == target_y)])
    #Scale data
    if def_scaler == None:
        scaler = preprocessing.StandardScaler().fit(training_data)
    else:
        scaler = def_scaler
    X_scaled = scaler.transform(training_data)
    return X_scaled, status, scaler

def main():
    training = pd.read_csv("LoanStatsNew.csv")

    X_scaled, status, scaler_init = prepare_data_for_year(training, 2009, def_scaler=None)
#X_scaled_2012, status_2012, scaler_2012 = prepare_data_for_year(training, 2012, def_scaler=None)
    X_scaled_2013, status_2013, _ = prepare_data_for_year(training, 2013, def_scaler=scaler_init)


#Train on a grid search for gamma and C
    parameters = [{'C': [0.1, 1, 10, 100, 1000],
                   'gamma': [0.1,0.01, 0.001],
                   'kernel': ['poly','rbf'], 'degree': [2, 3]}]
    classifier = grid_search.GridSearchCV(svm.SVC(C=1), parameters,
                                      verbose=3, n_jobs=4)
    classifier.fit(X_scaled, status)

#clf = svm.LinearSVC().fit(X_scaled, status)
#C = 100.0  # SVM regularization parameter
#rbf_svc = svm.SVC(kernel='poly', gamma=0.001, C=C).fit(X_scaled_2012,
#                                                      status_2012)



#Predict
    predict_2013 = classifier.predict(X_scaled_2013)
    print metrics.classification_report(status_2013, predict_2013)
#Report

#Cross validate
#scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
#Fit test data
    return 0

if __name__ == '__main__':
    main()
