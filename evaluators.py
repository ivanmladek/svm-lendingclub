from sklearn import (svm, preprocessing,
                     grid_search, metrics,
                     cross_validation)
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import (zero_one_loss,
                             recall_score)
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

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
    pl.show()


def forest_optim(X_scaled, status):
    """
    http://scikit-learn.org/0.11/auto_examples/ensemble/\
    plot_forest_importances.html#example-ensemble-plot-forest-importances-py
    """
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  compute_importances=True,
                                  random_state=0)
    forest.fit(X_scaled, status)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print "Feature ranking:"

    for f in xrange(len(X_scaled[0])):
        print "%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])

    # Plot the feature importances of the trees and of the forest
    import pylab as pl
    pl.figure()
    pl.title("Feature importances")

    for tree in forest.estimators_:
        pl.plot(xrange(len( tree.feature_importances_[indices])),
                tree.feature_importances_[indices], "r")

    pl.plot(xrange(len(importances[indices])),
            importances[indices], "b")
    pl.show()
