
"""
@author: ssunitha3

"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.neural_network import MLPClassifier
from util import plot_learning_curve,plot_validation_curve
from util import import_data
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time as time

class neuralNetwork(object):

    def main(self,X1,Y1,dataset_name,params):

        ## split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3, random_state=42)

        estimator = self.learning_curve(X1, Y1,params,dataset_name)
        tick = time.clock()
        estimator.fit(X_train, y_train)
        tock = time.clock() - tick
        print "Traning time for {} dataset".format(dataset_name), tock
        tick = time.clock()
        y_pred = estimator.predict(X_test)
        tock = time.clock() - tick
        print "Testing time for {} dataset".format(dataset_name), tock
        print "Accuracy Score {} dataset ".format(dataset_name), accuracy_score(y_test, y_pred)

        self.validation_curve(X1, Y1, params, dataset_name)

        ##Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        sns.heatmap(cm, center=True)
        #plt.show()
        plt.savefig('ConfusionMatrix for {} Dataset (neuralNetwork)'.format(dataset_name))



    def learning_curve(self,X1,Y1,param,dataset_name):

        title = "Learning Curve for {} Dataset (Neural Network)".format(dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)
        estimator = self.gridSearchCV(X_train, y_train, param,dataset_name)
        plot_learning_curve(estimator, title, X1, Y1, ylim=None, cv=5)
        plt.show()
        return estimator

    def gridSearchCV(self,X_train, y_train, param,dataset_name):

        pipeA = Pipeline([('Scale', StandardScaler()),
                          ('MLP', MLPClassifier(max_iter=5000, early_stopping=False, random_state=55))])

        gs = GridSearchCV(pipeA, param_grid=param, cv=5,n_jobs=-1)
        gs.fit(X_train, y_train)
        print "Best Params for {}".format(dataset_name),gs.best_params_
        print "Best Grid Scores for {}".format(dataset_name),gs.grid_scores_
        print "Best Estimator for {}".format(dataset_name),gs.best_estimator_
        return gs.best_estimator_

    def validation_curve(self,X1,Y1,param,dataset_name):

        alpha = param['MLP__alpha']
        title = "Validation Curve for {} Dataset (Neural Network)".format(dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)
        estimator = Pipeline([('Scale', StandardScaler()),
                          ('MLP', MLPClassifier(max_iter=2000, early_stopping=True, random_state=55))])
        plot_validation_curve(estimator,title,X1, Y1,"MLP__alpha",alpha,ylim=None,xlim=None, cv=5)

        plt.show()




if __name__ == '__main__':

    ###############################################################################

    X1, Y1, X2, Y2 = import_data()
    # alphas = np.logspace(-5, 3, 5)
    # d = X1.shape[1]
    # hiddens_letter = [(h, h) * l for l in [1, 2, 3] for h in [d, d // 2, d * 2]]
    # params_letter = {'MLP__activation': ['relu', 'logistic'], 'MLP__alpha': alphas,
    #                   'MLP__hidden_layer_sizes': hiddens_letter}
    # neuralNetwork().main(X1,Y1,"Letter Recognition",params_letter)
    # X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)

    ###############################################################################

    alphas = np.logspace(-5, 3, 5)
    d = X2.shape[1]
    d = d // (2 ** 4)
    hiddens_madelon = [(h, h) * l for l in [1, 2, 3] for h in [d, d // 2, d * 2]]
    params_madelon = {'MLP__activation': ['relu', 'logistic'], 'MLP__alpha': alphas,
                      'MLP__hidden_layer_sizes': hiddens_madelon}
    neuralNetwork().main(X2, Y2,"Madelon",params_madelon)