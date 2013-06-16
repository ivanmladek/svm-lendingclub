#!/export/disk0/wb/python2.6/bin/python
from sklearn import (svm, preprocessing,
                     grid_search, metrics,
                     cross_validation)

from sklearn.metrics import (zero_one_loss,
                             recall_score)
from operator import itemgetter
import re
import pandas as pd
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import urllib2
import numpy as np

import pdf as ppdf
from evaluators import rfe_optim, forest_optim


#TODO AUC score function
#Calculate KS score for 2007,2008,2009 loans


def check_funding(url):
    """
    Check realtime funding levels and only
    """
    try:
        text = urllib2.urlopen(url).read()
        float_str = text[text.index(' funded')-10:text.index(' funded')-1]
        float_str_clean = re.findall("\d+.\d+", float_str)
        return float(float_str_clean[0])
    except:
        return 0.0


def plot_histograms(finite, check_dictionary):
    for c in check_dictionary:
        try:
            plt.figure()
            plt.hist(finite[c][~np.isnan(finite[c])], bins=50)
            plt.title(c)
            plt.show()
        except:
            print c+' failed'

def binary_status(st):
    if (st == 'Issued' or
        st == 'Fully Paid' or
        st == 'Current' or
        st == 'Late (16-30 days)' or
        st == 'Does not meet the current credit policy.  Status:In Grace Period'
        ):
        return 0
    else:
        return 1

def parse_year(date):
    try:
        return datetime.strptime(date[0:10],'%Y-%m-%d').year
    except:
        return datetime.strptime(date[0:10],'%m-%d-%Y').year

def parse_finite(string):
    try:
        return np.nan_to_num(string)
    except:
        return 0.

def parse_percent(pc):
    try:
        return str(pc).replace("%","")
    except:
        return pc

PURPOSE_DICT = {
    'debt_consolidation':9,
    np.nan:7.,
    'educational':12.,
    'renewable_energy':13.,
    'car':6.,
    'medical':14.,
    'wedding':3.,
    'vacation':4.,
    'credit_card':10.,
    'other':7.,
    'moving':11.,
    'house':8.,
    'small_business':2.,
    'major_purchase':5.,
    'home_improvement':1.,
    'Home improvement':1.,
    'Business':2.,
    'Wedding expenses':3.,
    'Vacation':4.,
    'Major purchase':5.,
    'Car financing':6.,
    'Other':7.,
    'Home buying':8.,
    'Debt consolidation':9.,
    'Credit card refinancing':10.,
    'Moving and relocation':11.,
    }

CHECK_DICTIONARY = {
    'fico_range_high':[600.,850.],
    'annual_inc':[10000.,500000.],
    'loan_amnt':[100.,40000.],
    'dti':[0.,40.],
    'open_acc':[0.,60.],
    'total_acc':[0.,100.],
    'inq_last_6mths':[0.,15.],
    'revol_bal':[0.,200000.],
    'total_bal_ex_mort':[0.,5000000.],
    'num_accts_ever_120_pd':[0.,20.],
    'pub_rec_bankruptcies':[0.,5.],
    'tot_coll_amt':[0., 10000.],
    'num_rev_tl_bal_gt_0':[0.,40.],
    'total_rev_hi_lim':[0.,1000000.],
    'mo_sin_old_rev_tl_op':[0.,800.],
    'pub_rec_gt_100':[0.,10.],
    'delinq_2yrs':[0.,10.],
    }

def columns_both_training_predict(train, test):
    """
    Get a list of numerical columns common to both training
    and test dataset.
    """
    #TODO remove member_id
    float_train =[c for c,d in zip(train.columns,
                                   train.dtypes)
                  if (d == 'float64' or
                      d == 'int64')]
    float_test =[c for c,d in zip(test.columns,
                                  test.dtypes)
                 if  (d == 'float64' or
                      d == 'int64')]
    return set.intersection(
        set(float_train),
        set(float_test))

def check_validity(finite):
    """
    CHeck basic validity of all entries in the incoming dataset
    """
    for c in CHECK_DICTIONARY:
        check_array = np.array(finite[c][~np.isnan(finite[c])])
        out_of_boundc_c = np.flatnonzero((np.logical_and(
                check_array<= CHECK_DICTIONARY[c][0],
                check_array>= CHECK_DICTIONARY[c][1])))
        if len(out_of_boundc_c) == 0:
            print c+' within bounds'
        else:
            raise Exception(str(c+' out of bounds'))

def current_loan_parser(filename):
    """
    Current file is mis-formatted and will not load with pandas
    """
    f = open(filename,'r')
    lines = f.read().split("\n")
    f.close()
    entries = [lines[i].replace("\"","").split(",") for i in range(len(lines))]
    return entries

def train_test(X_scaled, status):
    #Train on a grid search for gamma and C
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
    ##Defaults are very uneven and thus we need to give them more
    ##weight, perform cross-validation
    classifier = grid_search.GridSearchCV(
        svm.SVC(C=1, probability=True),
        parameters,# zero_one_loss,
        cv=5,
        verbose=3, n_jobs=4,)
    print 'training'
    classifier.fit(X_scaled, status)
    print 'done training'
    return classifier

def predict_current(current_offer, features_to_train,
                    common_float_columns,
                    scaler_init, classifier,
                    year_train, update_current):
     #Predict current loan offer sheet
    #See

    offer_scaled, offer_status, _ = prepare_data_for_year(
        current_offer, [2013], common_float_columns,
        def_scaler= scaler_init)
    #current_loan
    predict_offer = classifier.predict(offer_scaled[:, features_to_train])
    predict_prob = classifier.predict_proba(offer_scaled[:, features_to_train])
    #Report
    print  year_train, "2013"
    print predict_offer
    current_info_prob=list()
    for i in range(len(predict_offer)):
        raw_str = ''
        for k in CHECK_DICTIONARY.keys():
            raw_str = raw_str+','+k+':'+str(current_offer[k][i])
        #Check loan is still open
        if update_current:
            funded_pcnt = check_funding(current_offer['url'][i])
        else:
            funded_pcnt = 0.
        if funded_pcnt < 100.:
            current_info_prob.append([predict_prob[i][1],
                                      predict_offer[i],
                                      current_offer['id'][i],
                                      current_offer['url'][i],
                                      current_offer['loan_amnt'][i],
                                      current_offer['funded_amnt'][i],
                                      current_offer['term'][i],
                                      current_offer['apr'][i],
                                      current_offer['purpose'][i],
                                      current_offer['review_status'][i],
                                      ])
    #Sort list according to probabilities
    curr_sorted = sorted(current_info_prob, key=itemgetter(0))

    #Print to a file
    best_count = 50
    f = open('predicted-best.csv','w')
    for c in curr_sorted[0:best_count]:
        f.write("%s\n" % ",".join(map(str,c)))
    f.close()
    f1 = open('predicted-worst.csv','w')
    for c in curr_sorted[-1*best_count:]:
        f1.write("%s\n" % ",".join(map(str,c)))
    f1.close()


    #Plot PDF
    pdf=ppdf.PDF()
    pdf.set_title('Lending Club Loan Applicant Ranking')
    pdf.set_author('SVM Risk Consulting')
    pdf.print_chapter(1,'RATING OF BORROWERS - BEST '+str(best_count),
                      'predicted-best.csv')
    pdf.print_chapter(len(curr_sorted) - best_count,
                      'RATING OF BORROWERS - WORST '+str(best_count),
                      'predicted-worst.csv')
    filename = 'SVM_Consulting_LendingClub_Ranking_'+datetime.now().date().strftime('%Y_%m_%d')+".pdf"
    pdf.output(filename,'F')

def prepare_data_for_year(training, target_y, float_columns,
                          def_scaler=None):
    #Weed out NaNs
    finite_ix = np.flatnonzero(np.isfinite([f for f in training.fico_range_high]))
    finite = training.take(finite_ix)

    #Prepare training and test data
    try:
        year_index = [i for i,d in enumerate(finite.list_d)
                      if (parse_year(d) in target_y)]
        status = np.array([binary_status(l) for l in finite.loan_status[year_index]])
    except:
        status = [np.nan]

    #Check validity of finite values
    check_validity(finite)
    #plot_histograms(finite, CHECK_DICTIONARY)


    #TODO Test with home ownership status,
    #emp_length,addr_state,mths_since_recent_inq
    #TODO Basically take all numeric values and run random forest
    #feature ranking on them
    #It is much easier to predict defaults with some payment data i.e.
    #not at loan origination but throughout the lifetime of the loan
    print finite
    print float_columns
    for i,f in enumerate(float_columns):
        print i,f
    #[2008] [2009]
    #          precision    recall  f1-score   support
    #      0       0.93      0.99      0.96      4147
    #      1       0.95      0.73      0.83      1159
    #avg / total   0.93      0.93      0.93      5306
    #[[4103   44]
    #[ 309  850]]
    training_data = np.nan_to_num(np.array(
            [finite[f] for f in float_columns])).transpose()[year_index,:]
    print training_data.shape
    print status

    print training_data[0]
    #Scale data
    if def_scaler == None:
        scaler = preprocessing.StandardScaler().fit(training_data)
    else:
        scaler = def_scaler
    X_scaled = scaler.transform(training_data)
    return X_scaled, status, scaler

def main(update_current=False):
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-t", "--train",
                      default=2008,)
    parser.add_option("-p", "--process",
                      default=2009)
    opts, args = parser.parse_args()

    training = pd.read_csv("LoanStatsNew.csv")
    #http://pandas.pydata.org/pandas-docs/stable/io.html#index-columns-and-trailing-delimiters
    #for trailing delimiters
    current_offer = pd.read_csv("InFunding2StatsNew.csv",
                    na_values=['\" \"','\"null\"'],
                    skiprows=1, delimiter=",",
                    index_col=False)
    year_train = eval(opts.train)
    year_predict = eval(opts.process)
    print year_train, year_predict
    common_float_columns = columns_both_training_predict(
        training, current_offer)

    X_scaled, status, scaler_init = prepare_data_for_year(training, year_train, common_float_columns, def_scaler=None)
    X_scaled_test, status_test, _ = prepare_data_for_year(training, year_predict, common_float_columns,  def_scaler=scaler_init)


    print "Random Forest optimization"
    features_to_train = forest_optim(X_scaled, status)
    #Perform RFE
    print "RFE optimization"
    rfe_optim(X_scaled, status)

    #Train
    classifier = train_test(X_scaled[:,features_to_train], status)
    print classifier

   #Predict
    predict_test = classifier.predict(X_scaled_test[:,features_to_train])

    #Report
    print year_train, year_predict
    print metrics.classification_report(status_test, predict_test)
    print metrics.confusion_matrix(status_test, predict_test)
    predict_current(current_offer,features_to_train, common_float_columns,
                    scaler_init, classifier, year_train,
                    update_current)

    return 0

if __name__ == '__main__':
    main()
