"""
@author: ssunitha3

"""

from sklearn import svm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import import_data
from sklearn.model_selection import  GridSearchCV, ShuffleSplit,StratifiedKFold
import numpy as np
from sklearn.svm import SVC
from util import plot_learning_curve
from util import import_data
from util import plot_validation_curve
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import time as time

class supportVectorMachine(object):

    def main(self, X1, Y1,dataset_name,kernel):

        # split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.33, random_state=42)
        classifier = self.learning_curve(X1, Y1, dataset_name,kernel)
        tick = time.clock()
        classifier.fit(X_train, y_train)
        tock = time.clock() - tick
        print "Traning time for {} dataset".format(dataset_name), tock
        tick = time.clock()
        y_pred = classifier.predict(X_test)

        tock = time.clock() - tick
        print "Testing time for {} dataset".format(dataset_name), tock

        print "Accuracy Score {} dataset ".format(dataset_name), accuracy_score(y_test, y_pred)
        self.validation_curve(X1, Y1,dataset_name)



    def learning_curve(self,X1,Y1,dataset_name,kernel):

        title = "Learning Curve for {} Dataset(supportVectorMachine)".format(dataset_name)
        cv = StratifiedKFold(n_splits=10,  random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)

        #  Linear Kernel
        if kernel == "linear":
            C_param1 = self.GridSearchCV1(X_train, y_train,dataset_name)
            estimator1 = Pipeline([('Scale', StandardScaler()),
                                   ('clf', LinearSVC(C=C_param1))])
            plot_learning_curve(estimator1, title, X1, Y1, ylim=None, cv=cv)

            return Pipeline([('Scale', StandardScaler()),
                             ('clf', LinearSVC(C=C_param1))])

        # RBF Kernel
        elif kernel == "rbf":
            C_param2 = self.GridSearchCV2(X_train, y_train, dataset_name)

            estimator1 = Pipeline([('Scale', StandardScaler()),
                                   ('clf', SVC(kernel='rbf',C=C_param2))])
            plot_learning_curve(estimator1, title, X1, Y1, ylim=None, cv=cv)

            return Pipeline([('Scale', StandardScaler()),
                             ('clf', SVC(kernel='rbf',C=C_param2))])
        #


    def validation_curve(self, X1, Y1,dataset_name):

        param_grid = [0.001, 0.01, 0.01, 1, 10, 100, 1000]
        title = "Validation Curve for {} Dataset (supportVectorMachine)".format(dataset_name)
        cv = StratifiedKFold(n_splits=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)
        estimator = Pipeline([('Scale', StandardScaler()),
                          ('clf', SVC(kernel='rbf'))])
        param_range = np.logspace(-6, -1, 5)
        plot_validation_curve(estimator, title, X1, Y1, "clf__C", param_grid, ylim=None, xlim=None, cv=cv)

        #plt.show()


    def GridSearchCV1(self,X_train, y_train,dataset_name):

        param_grid = { 'clf__C': [0.001,0.01,0.01,1, 10, 100, 1000]}

        pipeA = Pipeline([('Scale', StandardScaler()),
                          ('clf', LinearSVC())])
        clf = GridSearchCV(estimator=pipeA, param_grid=param_grid, cv=10)
        clf.fit(X_train, y_train)
        print "Best Linear SVC Params {} dataset ".format(dataset_name),clf.best_params_
        return clf.best_params_['clf__C']



    def GridSearchCV2(self,X_train, y_train,dataset_name):

        param_grid = {'clf__kernel': ['rbf'], 'clf__gamma': [1e-3, 1e-4],
                     'clf__C': [1, 10, 100, 1000]}
                  #  {'kernel': ['linear'], 'C': [0.001,0.01,0.01,1, 10, 100, 1000]}]

        pipeA = Pipeline([('Scale', StandardScaler()),
                          ('clf', SVC(kernel='rbf'))])
        clf = GridSearchCV(estimator=pipeA, param_grid=param_grid, cv=10)
        clf.fit(X_train, y_train)
        print "Best RBF Params {} dataset ".format(dataset_name),clf.best_params_
        return clf.best_params_['clf__C']




if __name__ == '__main__':

    X1, Y1, X2, Y2 = import_data()
    supportVectorMachine().main(X1, Y1,"Letter Recognition-Linear","linear")
    supportVectorMachine().main(X2, Y2,"Madelon-Linear","linear")
    supportVectorMachine().main(X1, Y1, "Letter Recognition-rbf", "rbf")
    supportVectorMachine().main(X2, Y2, "Madelon-rbf", "rbf")