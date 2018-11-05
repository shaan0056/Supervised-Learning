
"""
@author: ssunitha3

"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve,validation_curve
from sklearn.model_selection import ShuffleSplit
import pandas as pd


def import_data():

    data1 = pd.read_csv("LetterRecognition.csv")

    df = pd.read_csv('madelon_train.txt', sep=' ', header=None).drop(500,axis=1)
    #
    df_labels = pd.read_csv('madelon_trainlabels.txt', sep=' ', header=None)
    #
    df_test = pd.read_csv('madelon_valid.txt', sep=' ', header=None).drop(500, axis=1)
    #
    df_test_labels = pd.read_csv('madelon_validlabels.txt', sep=' ', header=None)



    X2 = pd.concat([df,df_test],axis=0)
    Y2 = pd.concat([df_labels, df_test_labels], axis=0)

    final = pd.concat([X2,Y2],axis =1)
    final.to_csv("Madelone_final.csv",sep=",",index=False)

    X1 = data1.values[:, 0:-1]
    Y1 = data1.values[:, -1]


    return X1, Y1, X2, Y2


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('{}.png'.format(title))
    plt.clf()

def plot_validation_curve(estimator,title, X, y, param_name,param_range,
                              ylim=None,xlim=None, cv=None):

        if ylim is not None:
            plt.ylim(*ylim)

        train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name,param_range=param_range,cv=cv, scoring="accuracy", n_jobs=1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.title(title)
        plt.xlabel(param_name)
        plt.ylabel("Score")
        #plt.xticks(xlim)
        lw = 2
        plt.semilogx(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.legend(loc="best")
        plt.savefig('{}.png'.format(title))
        plt.clf()

if __name__ == '__main__':

    import_data()