from sklearn.linear_model import *
import Preprocessing.preprocess as PP
import Classifier.processing as CL
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

WORLD_NEWS = 'data/Combined_WorldNews_DJIA.csv'
ECONOMICS = 'data/Combined_Economics_Saved.csv'
STOCKS = 'data/Combined_stocks_Saved.csv'
US_NEWS = 'data/Combined_news_Saved.csv'
TECH = 'data/Combined_technology_Saved.csv'
TEST = 'data/test.csv'

'''Calling Code'''
DAY_OFFSET = 0
DATA = WORLD_NEWS
# tranform_dates is ONLY needed for the Economics dataset
preprocess = PP.Preprocess(DATA, DAY_OFFSET, add_sentiment=False, stemming=False, add_date=False, transform_dates=False, add_relations=False)
results = preprocess.get_results()
data_train, data_test, y_train, y_test = results[0], results[1], results[2], results[3]

clf = LogisticRegression()
params1 = {
	"train_x": data_train,
	"train_y": y_train,
	"test_x": data_test,
	"test_y": y_test,
	"model": clf,
	"memento": 'data/demo.pkl'
}
CL.Classifier(params1)

clf = svm.SVC(kernel='linear')
params1 = {
	"train_x": data_train,
	"train_y": y_train,
	"test_x": data_test,
	"test_y": y_test,
	"model": clf,
	"memento": 'data/demo.pkl'
}
CL.Classifier(params1)

clf = SGDClassifier(shuffle=False, n_iter=1000, loss='squared_hinge')
params1 = {
	"train_x": data_train,
	"train_y": y_train,
	"test_x": data_test,
	"test_y": y_test,
	"model": clf,
	"memento": 'data/demo.pkl'
}
CL.Classifier(params1)
