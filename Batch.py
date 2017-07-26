import sys
from sklearn.linear_model import *
import Preprocessing.preprocess as PP
import Classifier.processing as CL
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

def print_config(config):
	print " ###################################### "
	print config

'''Calling Code'''
orig_stdout = sys.stdout
f = open('BATCH_OUTPUT.txt', 'w')
sys.stdout = f

WORLD_NEWS = 'data/Combined_WorldNews_DJIA.csv'
ECONOMICS = 'data/Combined_Economics_Saved.csv'
STOCKS = 'data/Combined_stocks_Saved.csv'
US_NEWS = 'data/Combined_news_Saved.csv'
TECH = 'data/Combined_technology_Saved.csv'
NY = 'data/Combined_NYT_Saved.csv'
DATA_SETS = [WORLD_NEWS, ECONOMICS, STOCKS, US_NEWS, TECH, NY]
OFFSETS = [-2, -1, 0, +1, +2]

for D in DATA_SETS:
	for O in OFFSETS:
		config1 = {
			"DATA": D,
			"DAY_OFFSET": O,
			"ADD_SENTIMENT": False,
			"STEMMING": False,
			"ADD_DATE": False,
			"ADD_RELATION": False
		}
		config2 = {
			"DATA": D,
			"DAY_OFFSET": O,
			"ADD_SENTIMENT": False,
			"STEMMING": True,
			"ADD_DATE": False,
			"ADD_RELATION": False
		}
		config3 = {
			"DATA": D,
			"DAY_OFFSET": O,
			"ADD_SENTIMENT": True,
			"STEMMING": False,
			"ADD_DATE": False,
			"ADD_RELATION": False
		}
		config4 = {
			"DATA": D,
			"DAY_OFFSET": O,
			"ADD_SENTIMENT": True,
			"STEMMING": True,
			"ADD_DATE": False,
			"ADD_RELATION": False
		}
		config5 = {
			"DATA": D,
			"DAY_OFFSET": O,
			"ADD_SENTIMENT": False,
			"STEMMING": False,
			"ADD_DATE": True,
			"ADD_RELATION": False
		}
		config6 = {
			"DATA": D,
			"DAY_OFFSET": O,
			"ADD_SENTIMENT": False,
			"STEMMING": False,
			"ADD_DATE": False,
			"ADD_RELATION": True
		}
		config7 = {
			"DATA": D,
			"DAY_OFFSET": O,
			"ADD_SENTIMENT": True,
			"STEMMING": True,
			"ADD_DATE": True,
			"ADD_RELATION": True
		}
		CONFIGS = [config1, config2, config3, config4, config5, config6, config7]
		for C in CONFIGS:
			print_config(C)
			preprocess = PP.Preprocess(C)
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


sys.stdout = orig_stdout
f.close()