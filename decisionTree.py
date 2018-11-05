"""
@author: ssunitha3

"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit,StratifiedKFold,StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from util import plot_learning_curve
from util import import_data
from util import plot_validation_curve
from sklearn.learning_curve import validation_curve
import time as time


class decisionTree(object):

    def main(self,X1,Y1,dataset_name):

        #########
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)


        classifier = self.learning_curve(X1,Y1,dataset_name)
        tick = time.clock()
        classifier.fit(X_train, y_train)
        tock = time.clock() - tick
        print "Traning time for {} dataset".format(dataset_name),tock
        tick = time.clock()
        y_pred = classifier.predict(X_test)

        tock = time.clock() - tick
        print "Testing time for {} dataset".format(dataset_name),tock

        print "Accuracy Score {} dataset ".format(dataset_name), accuracy_score(y_test, y_pred)

        self.validation_curve(X1, Y1,dataset_name)


    def GridSearchCV( self,X_train, y_train,dataset_name):

        param_grid = { 'max_depth': np.arange(1, 20)}
        tree = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=10)
        tree.fit(X_train, y_train)
        print "{} dataset Best Params".format(dataset_name),tree.best_params_
        return tree.best_params_['max_depth']

    def learning_curve(self,X1,Y1,dataset_name):

        title = "Learning Curve for {} Dataset (Decision Tree)".format(dataset_name)
        cv = StratifiedKFold(n_splits=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)
        max_depth = self.GridSearchCV(X_train, y_train,dataset_name)
        estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        plot_learning_curve(estimator, title, X1, Y1, ylim=None, cv=cv)

        plt.show()
        return DecisionTreeClassifier(max_depth=max_depth, random_state=42)

    def validation_curve(self,X1,Y1,dataset_name):

        param_grid = np.arange(1, 20)
        title = "Validation Curve for {} Dataset (Decision Tree)".format(dataset_name)
        cv = StratifiedKFold(n_splits=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)
        estimator = DecisionTreeClassifier(random_state=42)
        plot_validation_curve(estimator,title,X1, Y1,"max_depth",param_grid,ylim=None,xlim=None, cv=cv)

        plt.show()



if __name__ == '__main__':

    X1, Y1, X2, Y2 = import_data()

    decisionTree().main(X1, Y1,"Letter Recognition")
    decisionTree().main(X2,Y2,"Madelon")



