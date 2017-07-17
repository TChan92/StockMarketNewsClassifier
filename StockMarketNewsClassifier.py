from sklearn.linear_model import *
import Preprocessing.preprocess as PP
import Classifier.processing as CL
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import svm

preprocess = PP.Preprocess(add_sentiment=True, stemming=True)
results = preprocess.get_results()

'''Find the most important keys'''
# Most +ve keys = [912331, 441059, 849064, 689122, 913892, 393915, 791812, 799992, 63420, 403803]
# Most -ve keys = [88, 87, 67, 35, 34, 26, 9, 7, 6, 948294]]
# clf = svm.SVC(kernel='linear')
data_train, data_test, y_train, y_test = results[0], results[1], results[2], results[3]

clf = LogisticRegression()
# clf = MLPClassifier(hidden_layer_sizes=(90, 80, 70), solver='sgd')
# params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
# clf = ensemble.GradientBoostingRegressor(**params)
# clf = MLPClassifier(hidden_layer_sizes=(90, 80, 70), solver='sgd')
# clf = SGDClassifier(shuffle=False, n_iter=1000, loss='squared_hinge')

params1 = {"train_x": data_train, "train_y": y_train, "test_x": data_test, "test_y": y_test, "model": clf}
params1 = {
	"train_x": data_train,
	"train_y": y_train,
	"test_x": data_test,
	"test_y": y_test,
	"model": clf,
	"memento": 'data/demo.pkl'
}
lr = CL.Classifier(params1)


def demo():
	# Prompt for input data
	input_ = ''
	while True:
		input_ = raw_input("Enter HeadLine: ")
		if input_ == '':
			break
		else:
			head_line = input_
			trans_data = preprocess.transform_data(head_line)
			c = clf.predict(trans_data)
			# prob = clf.predict_proba(trans_data)
			result = ''

			if int(c[0]) == 0:
				result += "Down "
				# result += str(prob[0][0])
			else:
				result += "Up "
				# result += str(prob[0][1])
		print result


demo()
