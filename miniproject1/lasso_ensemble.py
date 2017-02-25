import csv
import time
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

def normalize(x):
    '''This function nomalizes each columns of the input 2d array.'''
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_std[x_std == 0] = 1
    x1 = (x - x_mean) / x_std
    return x1

def selectFeature(X_train, y_train, X_test1, X_test2, alpha1):
    '''This function select the features of normalized data (i.e., np.std(X[:,j]) = 1 or 0).
    Firstly Lasso regression will used to fit the data 'X_train, y_train', the column X[:,j]
    will be selected if wj > 0.'''
    lasso = linear_model.Lasso(alpha=alpha1)
    lasso.fit(X_train, y_train)
    cols = (abs(lasso.coef_) > 0)
    x_train = X_train[:, cols]
    x_test1 = X_test1[:, cols]
    x_test2 = X_test2[:, cols]
    str1 = 'The regularization alpha in the Lasso regression is %.6f ; '
    str2 = ' %d features have been selected.'
    print(str1 % alpha1, str2 % sum(cols))
    return (x_train, x_test1, x_test2)

start = time.time()

# load the the data from the files
with open('train_2008.csv', 'r') as file1: 
    lines1 = csv.reader(file1, delimiter=',', quotechar='|') 
    next(lines1, None)
    data1 = np.array([line for line in lines1], dtype=float)

with open('test_2008.csv', 'r') as file2:
	lines2 = csv.reader(file2, delimiter=',', quotechar='"')
	next(lines2, None)
	data2 = np.array([line for line in lines2], dtype=float)

with open('test_2012.csv', 'r') as file3:
	lines3 = csv.reader(file3, delimiter=',', quotechar='"')
	next(lines3, None)
	data3 = np.array([line for line in lines3], dtype=float)

# convert the data to float numpy array 
N_train = len(data1)
y_train = 2 * (data1[:, -1] - 1.5)  # maps 1 to -1, 2 to 1
X_train = normalize(data1[:, :-1])
X_train[:, 0] = 1
X_test1 = normalize(data2)
X_test1[:, 0] = 1
X_test2 = normalize(data3)
X_test2[:, 0] = 1
alpha1 = 0.0075
X_train, X_test1, X_test2 = selectFeature(X_train, y_train, X_test1, X_test2, alpha1) 
d = len(X_train[0])


# train the model and calculate the scores by cross-validation
N = 200
clf1 = AdaBoostClassifier(n_estimators=N)
clf2 = GradientBoostingClassifier(n_estimators=N, max_depth=4)
clf3 = RandomForestClassifier(n_estimators=N, min_samples_split=12)
eclf = VotingClassifier(estimators=[('ab', clf1), ('gb', clf2), ('rf', clf3)], voting='hard')
clf_lst = [clf1, clf2, clf3, eclf]
name_lst = ['AdaBoost', 'GradientBoost', 'RandomForest', 'Ensemble']
for clf, name in zip(clf_lst, name_lst):
    scores = cross_val_score(clf, X_train, y_train, cv=2, scoring='accuracy')
    print("Accuracy: %0.6f (+/- %0.6f) [%s]" % (scores.mean(), scores.std(), name))

# write the prediction data into the submission file
eclf.fit(X_train, y_train)
y_test1 = eclf.predict(X_test1)
print([sum(y_test1==-1), sum(y_test1==1)])
with open('submission2008.csv', 'w', newline='') as file: 
	filewriter = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	filewriter.writerow(['id', 'PES1'])
	for i, yi in enumerate(y_test1):
		filewriter.writerow([str(i), str(int(yi/2 + 1.5))])
y_test2 = eclf.predict(X_test2)
print([sum(y_test2==-1), sum(y_test2==1)])
with open('submission2012.csv', 'w', newline='') as file: 
	filewriter = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	filewriter.writerow(['id', 'PES1'])
	for i, yi in enumerate(y_test2):
		filewriter.writerow([str(i), str(int(yi/2 + 1.5))])


# print running time
stop = time.time()
print('The running time is ', stop - start)



