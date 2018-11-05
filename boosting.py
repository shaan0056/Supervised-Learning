"""
@author: ssunitha3

"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import  GridSearchCV, ShuffleSplit, StratifiedKFold
import matplotlib.pyplot as plt
from util import plot_learning_curve
from util import plot_validation_curve
from util import import_data
import numpy as np
import time as time

class boosting(object):

    def main(self,X1,Y1,dataset_name):
        # split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3, random_state=42)

        if dataset_name == "Letter Recognition":
            d = 19
        if dataset_name == "Madelon":
            d = 6
        classifier = self.learning_curve(X1, Y1, dataset_name,d)
        tick = time.clock()
        classifier.fit(X_train, y_train)
        tock = time.clock() - tick
        print "Traning time for {} dataset".format(dataset_name), tock
        tick = time.clock()
        y_pred = classifier.predict(X_test)

        tock = time.clock() - tick
        print "Testing time for {} dataset".format(dataset_name), tock

        print "Accuracy Score {} dataset ".format(dataset_name), accuracy_score(y_test, y_pred)
        self.validation_curve(X1, Y1,dataset_name,d)



    def GridSearchCV( self,X_train, y_train,dataset_name,d):
        param_grid = { 'n_estimators': [1,2,5,10,20,30,40,50,60,70,80,90,100] }

        tree = GridSearchCV(AdaBoostClassifier( base_estimator = DecisionTreeClassifier(max_depth=d, random_state=100, min_samples_leaf=1), learning_rate=0.5), param_grid=param_grid, cv=10)
        tree.fit(X_train, y_train)
        print "Best Params {} dataset ".format(dataset_name),tree.best_params_
        return tree.best_params_['n_estimators']

    def learning_curve(self,X1,Y1,dataset_name,d):

        title = "Learning Curve for {} Dataset (Boosting)".format(dataset_name)
        cv = StratifiedKFold(n_splits=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)
        n_estimators = self.GridSearchCV(X_train, y_train,dataset_name,d)
        estimator = AdaBoostClassifier( base_estimator = DecisionTreeClassifier(max_depth=d, random_state=42),n_estimators=n_estimators)
        plot_learning_curve(estimator, title, X1, Y1, ylim=None, cv=cv)
        return  AdaBoostClassifier( base_estimator = DecisionTreeClassifier(max_depth=d, random_state=42),n_estimators=n_estimators)
        plt.show()

    def validation_curve(self,X1,Y1,dataset_name,d):

        param_grid = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        title = "Validation Curve for {} Dataset (Boosting)".format(dataset_name)
        cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)
        estimator = AdaBoostClassifier( base_estimator = DecisionTreeClassifier(max_depth=d, random_state=42))
        plot_validation_curve(estimator,title,X1, Y1,"n_estimators",param_grid,ylim=None,xlim=None, cv=cv)

        plt.show()


if __name__ == '__main__':

    X1, Y1, X2, Y2 = import_data()
    boosting().main(X1,Y1,"Letter Recognition")
    boosting().main(X2,Y2,"Madelon")

