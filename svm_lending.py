#!/export/disk0/wb/python2.6/bin/python
from sklearn import (svm, preprocessing,
                     grid_search, metrics,
                     cross_validation)
from sklearn.metrics import (zero_one_loss,
                             recall_score)
import re
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

def parse_year(date):
    try:
        return datetime.strptime(date[0:10],'%Y-%m-%d').year
    except:
        return datetime.strptime(date[0:10],'%m-%d-%Y').year

def parse_percent(pc):
    try:
        return str(pc).replace("%","")
    except:
        return pc

def prepare_data_for_year(training, target_y, def_scaler=None):
    #Weed out NaNs
    finite_ix = np.flatnonzero(np.isfinite([f for f in training.fico_range_high]))
    finite = training.take(finite_ix)

    #Prepare training and test data
    try:
        status = np.array([binary_status(l) for l,d in zip(finite.loan_status,
                                                           finite.list_d)
                           if (datetime.strptime(d,'%Y-%m-%d').year in target_y)])
    except:
        status = [np.nan]
    print finite
    training_data = np.array([[len(str(desc)),
                               f, ann_inc,
                               amount, dti,
                               open_acc, total_acc,
                               num_inq,
                               float(revol_bal),
                               np.nan_to_num(float(parse_percent(revol_util))),
                               float(parse_percent(apr)),
                               np.nan_to_num(total_balance),
                               np.nan_to_num(default120),
                               np.nan_to_num(bankruptcies),
                               np.nan_to_num(tot_coll_amnt),
                               np.nan_to_num(rev_gt0),
                               np.nan_to_num(rev_hilimit),
                               np.nan_to_num(oldest_rev),
                               np.nan_to_num(pub_rec),
                               np.nan_to_num(delinq_2)]
                              for desc, f, ann_inc,amount,dti,
                              open_acc,total_acc, num_inq, revol_bal,revol_util, apr,
                              emp_length,
                              total_balance,default120,
                              bankruptcies,
                              tot_coll_amnt,
                              rev_gt0,
                              rev_hilimit,
                              oldest_rev,
                              pub_rec,
                              delinq_2,
                              list_d
                              in zip(finite.desc,
                                     finite.fico_range_high,
                                     finite.annual_inc,
                                     finite.loan_amnt,
                                     finite.dti,
                                     finite.open_acc,
                                     finite.total_acc,
                                     finite.inq_last_6mths,
                                     finite.revol_bal,
                                     finite.revol_util,
                                     finite.apr,
                                     finite.emp_length,
                                     finite.total_bal_ex_mort,
                                     finite.num_accts_ever_120_pd,
                                     finite.pub_rec_bankruptcies,
                                     finite.tot_coll_amt,
                                     finite.num_rev_tl_bal_gt_0,
                                     finite.total_rev_hi_lim,
                                     finite.mo_sin_old_rev_tl_op,
                                     finite.pub_rec_gt_100,
                                     finite.delinq_2yrs,
                                     finite.list_d,)
                              if ( parse_year(list_d) in target_y)])
    #Scale data
    if def_scaler == None:
        scaler = preprocessing.StandardScaler().fit(training_data)
    else:
        scaler = def_scaler
    X_scaled = scaler.transform(training_data)
    return X_scaled, status, scaler

def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-t", "--train",
                      default=2008,)
    parser.add_option("-p", "--process",
                      default=2009)
    opts, args = parser.parse_args()

    training = pd.read_csv("LoanStatsNew.csv")
    print training.columns
    #print [training[t][70000:70100]  for t in training.columns]

    year_train = eval(opts.train)
    year_predict = eval(opts.process)
    print year_train, year_predict
    X_scaled, status, scaler_init = prepare_data_for_year(training, year_train, def_scaler=None)
    X_scaled_test, status_test, _ = prepare_data_for_year(training, year_predict, def_scaler=scaler_init)


#Train on a grid search for gamma and C
    parameters = [{'C': [0.001, 0.01, 0.1, 1, 10,],# 100, 1000],
                   'gamma': [0.1, 0.01,  0.001, 0.0001],
                   'kernel': ['poly','rbf'], 'degree': [2],
                   }]
    ##Defaults are very uneven and thus we need to give them more
    ##weight, perform cross-validation
    classifier = grid_search.GridSearchCV(
        svm.SVC(C=1, class_weight ={-1: 1, 1: 5}),
        parameters,zero_one_loss,
        verbose=1, n_jobs=4,)
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
    print 'Cross-validating'
    scores = cross_validation.cross_val_score(
        classifier, X_scaled, status, cv=5)
    print scores

    #Predict current loan offer sheet
    #current_offer = pd.read_csv("InFunding2StatsNew.csv")
    #print current_offer
    #offer_scaled, offer_status, _ = prepare_data_for_year(current_offer, 2013, def_scaler= scaler_init)
    #predict_offer = classifier.predict(offer_scaled)
    #Report
    #print year_train, "2013"
    #print predict_offer

    return 0

if __name__ == '__main__':
    main()
