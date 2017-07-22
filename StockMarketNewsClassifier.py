from sklearn.linear_model import *
import Preprocessing.preprocess as PP
import Classifier.processing as CL
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

preprocess = PP.Preprocess(add_sentiment=False, stemming=False)
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


# demo()
