from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.externals import joblib

class Classifier():
	def __init__(self, params):
		self._model = params["model"] 	# Classifier
		self._train_x = params["train_x"]
		self._train_y = params["train_y"]
		self._test_x = params["test_x"]
		self._test_y = params["test_y"]
		self._model = self._model.fit(self._train_x, self._train_y)
		joblib.dump(self._model, params["memento"])
		joblib.dump(self._test_x, 'data/testx.pkl')
		joblib.dump(self._test_y, 'data/testy.pkl')

		predictions = self._model.predict(self._test_x)
		cm = confusion_matrix(self._test_y, predictions)
		print (cm)
		print "Accuracy " + str(accuracy_score(self._test_y, predictions))
		print(classification_report(self._test_y, predictions))

# Grab the data
# data = pd.read_csv('../data/Combined_News_DJIA.csv')
# train = data[data['Date'] < '2015-01-01']
# test = data[data['Date'] > '2014-12-31']

# params1 = {"train": train, "test": test, "model": LogisticRegression()}
# lr = Classifier(params1)
#
# params2 = {"train": train, "test": test, "model" : MLPClassifier(hidden_layer_sizes=(90, 80, 70))}
# lr2 = Classifier(params2)
#
# params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01, 'loss': 'ls'}
# clf = ensemble.GradientBoostingRegressor(**params)
# params3 = {"train": train, "test": test, "model" : clf}
# lr3 = Classifier(params3)