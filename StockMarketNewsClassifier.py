from sklearn.linear_model import *
import Preprocessing.preprocess as PP
import Classifier.processing as CL
from sklearn.linear_model import LogisticRegression


preprocess = PP.Preprocess()
results = preprocess.get_results()

data_train, data_test, y_train, y_test = results[0], results[1], results[2], results[3]
clf = SGDClassifier(n_iter=100)
clf.set_params()
params1 = {"train_x": data_train, "train_y": y_train, "test_x": data_test, "test_y": y_test, "model": clf}
lr = CL.Classifier(params1)