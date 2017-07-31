from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np


class Classifier():
	def __init__(self, params):
		self._model = params["model"]
		self._train_x = params["train_x"]
		self._train_y = params["train_y"]
		self._test_x = params["test_x"]
		self._test_y = params["test_y"]
		self._model = self._model.fit(self._train_x, self._train_y)
		predictions = self._model.predict(self._test_x)
		cm = confusion_matrix(self._test_y, predictions)
		print (cm)
		print "Accuracy " + str(accuracy_score(self._test_y, predictions))
		print(classification_report(self._test_y, predictions))
