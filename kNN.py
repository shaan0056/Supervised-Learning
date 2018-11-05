
"""
@author: ssunitha3

"""
#
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import  GridSearchCV, ShuffleSplit, StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from util import plot_learning_curve
from util import plot_validation_curve
from util import import_data
import time as time

class kNN(object):

    def main(self,X1,Y1,dataset_name):


        # split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3, random_state=42)

        classifier = self.learning_curve(X1, Y1,dataset_name)

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



    def GridSearchCV(self,X_train, y_train,dataset_name):

        myList = list(range(1, 50))
        neighbors = filter(lambda x: x % 2 != 0, myList)
        param_grid = {'n_neighbors': neighbors}

        tree = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=10)
        tree.fit(X_train, y_train)
        print "Best Params {} dataset".format(dataset_name),tree.best_params_
        return tree.best_params_['n_neighbors']

    def learning_curve(self,X1,Y1,dataset_name):

        title = "Learning Curve for {} Dataset(KNN)".format(dataset_name)
        cv = StratifiedKFold(n_splits=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)
        n_neighbors = self.GridSearchCV(X_train, y_train,dataset_name)

        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
        plot_learning_curve(estimator, title, X1, Y1, ylim=None, cv=cv)

        plt.show()
        return KNeighborsClassifier(n_neighbors=n_neighbors)

    def validation_curve(self,X1,Y1,dataset_name):

        myList = list(range(1, 50))
        neighbors = filter(lambda x: x % 2 != 0, myList)
        param_grid = neighbors
        title = "Validation Curve for {} Dataset (KNN)".format(dataset_name)
        cv = StratifiedKFold(n_splits=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)
        estimator = KNeighborsClassifier()
        plot_validation_curve(estimator,title,X1, Y1,"n_neighbors",param_grid,ylim=None,xlim=(1,50), cv=cv)

        plt.show()



if __name__ == '__main__':

    X1, Y1, X2, Y2 = import_data()
    kNN().main(X1,Y1,"Letter Recognition")
    kNN().main(X2,Y2,"Madelon")