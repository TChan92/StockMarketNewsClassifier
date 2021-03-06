from sklearn.linear_model import *
import Preprocessing.preprocess as PP
import Classifier.processing as CL
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

WORLD_NEWS = 'Data/Combined_WorldNews_DJIA.csv'
ECONOMICS = 'Data/Combined_Economics_Saved.csv'
STOCKS = 'Data/Combined_stocks_Saved.csv'
US_NEWS = 'Data/Combined_news_Saved.csv'
TECH = 'Data/Combined_technology_Saved.csv'
TEST = 'Data/test.csv'
NY = 'Data/Combined_NYT_Saved.csv'

'''Calling Code'''
config = {
	"DATA": WORLD_NEWS,
	"DAY_OFFSET": -1,
	"ADD_SENTIMENT": False,
	"STEMMING": False,
	"ADD_DATE": True,
	"ADD_RELATION": True
}
preprocess = PP.Preprocess(config)
results = preprocess.get_results()
data_train, data_test, y_train, y_test = results[0], results[1], results[2], results[3]

clf = LogisticRegression()
params1 = {
	"train_x": data_train,
	"train_y": y_train,
	"test_x": data_test,
	"test_y": y_test,
	"model": clf,
	"memento": 'Data/demo.pkl'
}
CL.Classifier(params1)

clf = svm.SVC(kernel='linear')
params1 = {
	"train_x": data_train,
	"train_y": y_train,
	"test_x": data_test,
	"test_y": y_test,
	"model": clf,
	"memento": 'Data/demo.pkl'
}
CL.Classifier(params1)

clf = SGDClassifier(shuffle=False, n_iter=1000, loss='squared_hinge')
params1 = {
	"train_x": data_train,
	"train_y": y_train,
	"test_x": data_test,
	"test_y": y_test,
	"model": clf,
	"memento": 'Data/demo.pkl'
}
CL.Classifier(params1)
