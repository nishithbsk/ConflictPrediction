import numpy as np
from sklearn.linear_model import LogisticRegression

def load_data(data_path='../../data/processed/logistic_regression/uganda_data.npy'):
    # returns tuple (X_train, X_test, y_train, y_test)
    return np.load(data_path)

X_train, X_test, y_train, y_test = load_data()


model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print "Aligned:", model.score(X_test, y_test)

precision_num, precision_denom = 0.0, 0.0
for i in range(len(pred)):
	if y_test[i] == 1:
		if pred[i] == 1: 
			precision_num += 1
			precision_denom += 1
	else:
		if pred[i] == 1: precision_denom += 1
print "Precision:", float(precision_num)/precision_denom

recall_num, recall_denom = 0.0, 0.0
for i in range(len(pred)):
	if y_test[i] == 1:
		if pred[i] == 1: 
			recall_num += 1
			recall_denom += 1
		else:
			recall_denom += 1

print "Recall:", float(recall_num)/recall_denom

