#!/export/disk0/wb/python2.6/bin/python
from sklearn import (svm, preprocessing,
                     grid_search, metrics)
import pandas as pd

from datetime import datetime
import numpy as np

def binary_status(st):
    if (st == 'Issued' or
        st == 'Fully Paid' or
        st == 'Current' #or
        #st == 'Late (16-30 days)' or
        #st == 'Does not meet the current credit policy.  Status:In Grace Period'
        ):
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
                               amount, dti, open_acc, num_inq,
                               float(revol_bal),
                               np.nan_to_num(float(str(revol_util).replace("%",""))),
                               float(apr.replace("%","")),
                               np.nan_to_num(total_balance),
                               np.nan_to_num(default120)]
                              for f, ann_inc,amount,dti,
                              open_acc,num_inq, revol_bal,revol_util, apr,
                              total_balance,default120,
                              issue_d
                              in zip(finite.fico_range_high,
                                     finite.annual_inc,
                                     finite.loan_amnt,
                                     finite.dti,
                                     finite.open_acc,
                                     finite.inq_last_6mths,
                                     finite.revol_bal,
                                     finite.revol_util,
                                     finite.apr,
                                     finite.total_bal_ex_mort,
                                     finite.num_accts_ever_120_pd,
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
    print training.columns
    print [training[t][:10]  for t in training.columns]

    year_train = 2008
    year_predict = 2009
    X_scaled, status, scaler_init = prepare_data_for_year(training, year_train, def_scaler=None)
    X_scaled_test, status_test, _ = prepare_data_for_year(training, year_predict, def_scaler=scaler_init)


#Train on a grid search for gamma and C
    parameters = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                   'gamma': [0.1, 0.01,  0.001, 0.0001],
                   'kernel': ['poly','rbf'], 'degree': [2],
                   }]
    ##Defaults are very uneven and thus we need to give them more weight
    classifier = grid_search.GridSearchCV(
        svm.SVC(C=1, class_weight ={-1:1, 1: 1}),
        parameters, verbose=3, n_jobs=4,)
    print 'training'
    classifier.fit(X_scaled, status)
    print 'done training'
    print classifier

#Predict
    predict_test = classifier.predict(X_scaled_test)

#Report
    print year_train, year_predict
    print metrics.classification_report(status_test, predict_test)
    print metrics.confusion_matrix(status_test, predict_test)
#Cross validate
#scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
#Fit test data
    return 0

if __name__ == '__main__':
    main()
