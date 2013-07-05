#!/export/disk0/wb/python2.6/bin/python
from sklearn import (svm, preprocessing,
                     grid_search, metrics,
                     cross_validation)

from sklearn.metrics import (zero_one_loss,
                             recall_score)
from StringIO import StringIO
from operator import itemgetter
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import urllib2
import numpy as np

import pdf as ppdf
from evaluators import (rfe_optim, forest_optim, roc,
                        parameters)
import geocode

#TODO AUC score function
#Calculate KS score for 2007,2008,2009 loans
# TODO for each run output details into a txt with
#columns used, model details and confusion matrix

def download_current():
    url = "https://www.lendingclub.com/fileDownload.action?file=InFunding2StatsNew.csv&type=gen"
    print 'Downloading current offers from: '+url
    text = urllib2.urlopen(url).read()
    return text

def download_portfolio():
    url = "https://www.lendingclub.com/fileDownload.action?file=LoanStatsNew.csv&type=gen"
    print 'Downloading portfolio from: '+url
    text = urllib2.urlopen(url).read()
    return text



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
        st == 'Current' #or
        #st == 'Late (16-30 days)' or
        #st == 'Does not meet the current credit policy.  Status:In Grace Period'
        ):
        return 0
    if st == np.NaN:
        raise Exception("No NaNs allowed in status")
    else:
        return 1

def parse_year(date):
    try:
        return datetime.strptime(date[0:10],'%Y-%m-%d').year
    except:
        return 2010#datetime.strptime(date[0:10],'%m-%d-%Y').year

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

def convert_columns_float(train):
    float_train = list()
    for c in train.columns:
        try:
            train[c].astype('float')
            float_train.append(c)
        except:
            pass
            #print 'train can not convert: '+c
    return float_train

def columns_both_training_predict(train, test):
    """
    Get a list of numerical columns common to both training
    and test dataset.
    """
    #TODO remove member_id
    float_train = convert_columns_float(train)
    float_test = convert_columns_float(test)
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
    #TODO Try more classifiers
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


def trailing_delimiter_parser(filename):
    current_offer = pd.read_csv(filename,
                                na_values=['\" \"','\"null\"'],
                                skiprows=1, delimiter=",",
                                index_col=False)
    return current_offer

def standard_parser(filename):
    return pd.read_csv(filename)

parser_options = {"InFun": trailing_delimiter_parser,
                  "LoanS": standard_parser,}



class SVMLending():
    """
    Training and preciting class for SVM credit scoring of Lending Club loans.
    """

    def __init__(self, training, current_offer, year_train, year_predict):
        print year_train, year_predict
        self.common_float_columns = columns_both_training_predict(
        training, current_offer)
        print self.common_float_columns
        print training, current_offer
        #Base training data
        self.X_scaled, self.status, self.scaler_init = \
            self.prepare_data_for_year(training, year_train,
                                       self.common_float_columns,
                                       def_scaler=None)
        #Out of sample test data
        self.X_scaled_test, self.status_test, _ = \
            self.prepare_data_for_year(training, year_predict,
                                       self.common_float_columns,
                                       def_scaler=self.scaler_init)

    def prepare_data_for_year(self, training, target_y, float_columns,
                              def_scaler=None):
        #Weed out NaNs
        #finite_ix = np.flatnonzero(np.isfinite([f for f in training.fico_range_high]))
        #finite = training.take(finite_ix)

        #Prepare training and test data
        year_index = [i for i,d in enumerate(training.list_d)
                      if (parse_year(d) in target_y)]
        try:
            #print np.array([l for l in training.loan_status])
            raw_status = training.loan_status
            status = np.array([binary_status(r) for i,r in enumerate(raw_status)
                               if i in year_index])
            print status
            #status = np.array([binary_status(l) for l in training.loan_status[year_index]])
        except:
            print 'No status data'
            status = [np.nan]

        #Convert float_columns to float
        finite_float = training[list(float_columns)].astype('float')

        #Check validity of finite values
        check_validity(finite_float)
        #plot_histograms(finite, CHECK_DICTIONARY)
        print finite_float
        for i,f in enumerate(float_columns):
            training_data = np.nan_to_num(np.array(
                    [finite_float[f] for f in
                     float_columns])).transpose()[year_index,:]
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

    def train(self, X_scaled, status, tol= 0.02):
        print "Random Forest optimization"
        features_to_train = forest_optim(X_scaled, status, FEAT_TOL=tol)
        print features_to_train
        #Perform RFE
        #print "RFE optimization"
        #n_feat_optimal = rfe_optim(X_scaled, status)
        #print 'optimal features'
        #print features_to_train[0:n_feat_optimal]

        #Train
        classifier = train_test(X_scaled[:,features_to_train], status)
        print classifier.best_estimator_
        return classifier, features_to_train

    def predict(self, classifier, X_scaled_test,
                features_to_train, status_test):
        #Predict
        predict_test = classifier.best_estimator_.predict(X_scaled_test[:,features_to_train])

        #Report
        print metrics.classification_report(status_test, predict_test)
        print metrics.confusion_matrix(status_test, predict_test)
        print 'ROC computation'
        roc(X_scaled_test[:,features_to_train], status_test,
            classifier.best_estimator_)
        return 0

    def predict_current(self, current_offer, features_to_train,
                        common_float_columns,
                        classifier, update_current=False):

        offer_scaled, offer_status,_ = \
            self.prepare_data_for_year(current_offer, [2013],
                                       common_float_columns,
                                       def_scaler=self.scaler_init)

        #current_loan
        predict_offer = classifier.predict(offer_scaled[:, features_to_train])
        predict_prob = classifier.predict_proba(offer_scaled[:, features_to_train])


        #TODO separate Report fn
        print predict_offer
        print predict_prob

        current_info_prob=list()
        for i in range(len(predict_offer)):
            #Check loan is still open
            if update_current:
                funded_pcnt = check_funding(current_offer['url'][i])
            else:
                funded_pcnt = 0.
            if funded_pcnt < 100.:
                current_info_prob.append([predict_prob[i][1],
                                          predict_offer[i],
                                          np.asarray(current_offer['id'])[i],
                                          np.asarray(current_offer['url'])[i],
                                          np.asarray(current_offer['loan_amnt'])[i],
                                          np.asarray(current_offer['funded_amnt'])[i],
                                          np.asarray(current_offer['term'])[i],
                                          np.asarray(current_offer['apr'])[i],
                                          np.asarray(current_offer['purpose'])[i],
                                          np.asarray(current_offer['latitude'])[i],
                                          np.asarray(current_offer['longitude'])[i],
                                          #current_offer['review_status'][i],
                                          ])
        #Sort list according to probabilities
        curr_sorted = sorted(current_info_prob, key=itemgetter(0))

        #Print to a file
        best_count = 5000
        f = open('predicted-best.csv','w')
        for c in curr_sorted[0:best_count]:
            f.write("%s\n" % ",".join(map(str,c)))
        f.close()
        f1 = open('predicted-worst.csv','w')
        for c in curr_sorted[-1*best_count:]:
            f1.write("%s\n" % ",".join(map(str,c)))
        f1.close()


        #TODO separate Plot PDF
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
        return 0


def main(update_current=False):
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-t", "--train",
                      default='[2008]',)
    parser.add_option("-p", "--process",
                      default='[2009]')
    parser.add_option("-f", "--train_file",
                      default="LoanStatsNew.csv")
    parser.add_option("-g", "--geocode",
                      default='False')
    parser.add_option("-i", "--test_file",
                      default="InFunding2StatsNew.csv")
    parser.add_option("-l", "--tol",
                      default=0.05)
    opts, args = parser.parse_args()
    #Interpret raining years
    year_train = eval(opts.train)
    year_predict = eval(opts.process)

    ###############################
    #Read and geocode training data
    g = geocode.Geocode()
    geo = eval(opts.geocode)
    training = g.process_file(opts.train_file, geocode=geo)

    #Download current offer
    current_offer = g.process_file(StringIO(download_current()),
                                   geocode=geo)

    ###############################
    #Train
    LC = SVMLending(training, current_offer, year_train,
                    year_predict)
    classifier, features_to_train = LC.train(LC.X_scaled, LC.status,
                                             tol=opts.tol)

    #Predict out-of-sample data
    LC.predict(classifier, LC.X_scaled_test,
               features_to_train, LC.status_test)

    ################################
    #Predict current_offering
    LC.predict_current(current_offer, features_to_train,
                       LC.common_float_columns,
                       classifier, update_current=False)
    return 0

if __name__ == '__main__':
    main()
