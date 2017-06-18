from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.externals import joblib

clf = joblib.load('data/grad.pkl')
X = joblib.load('data/testx.pkl')
y = joblib.load('data/testy.pkl')
predictions = clf.predict(X)
cm = confusion_matrix(y, predictions)
print (cm)
print "Accuracy " + str(accuracy_score(y, predictions))
print(classification_report(y, predictions))




