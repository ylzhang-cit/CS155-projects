import csv
import time
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier

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
N_half = int(N_train / 2)
y_train = 2 * (data1[:, -1] - 1.5)  # maps 1 to -1, 2 to 1
X_train = normalize(data1[:, :-1])
X_train[:, 0] = 1
X_test1 = normalize(data2)
X_test1[:, 0] = 1
X_test2 = normalize(data3)
X_test2[:, 0] = 1
alpha1 = 0.0015
X_train, X_test1, X_test2 = selectFeature(X_train, y_train, X_test1, X_test2, alpha1) 
d = len(X_train[0])


# train the model and calculate the scores by cross-validation
N = 200
clf1 = AdaBoostClassifier(n_estimators=N)
clf1.fit(X_train[:N_half], y_train[:N_half])
score1 = clf1.score(X_train[N_half:], y_train[N_half:])
clf2 = AdaBoostClassifier(n_estimators=N)
clf2.fit(X_train[N_half:], y_train[N_half:])
score2 = clf2.score(X_train[:N_half], y_train[:N_half])
print("The cross-validation scores are : ", score1, score2)


# write the prediction data into the submission file
clf = AdaBoostClassifier(n_estimators=N)
clf.fit(X_train, y_train)
y_test1 = clf2.predict(X_test1)
print([sum(y_test1==-1), sum(y_test1==1)])
with open('submission2008.csv', 'w', newline='') as file: 
	filewriter = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	filewriter.writerow(['id', 'PES1'])
	for i, yi in enumerate(y_test1):
		filewriter.writerow([str(i), str(int(yi/2 + 1.5))])
y_test2 = clf2.predict(X_test2)
print([sum(y_test2==-1), sum(y_test2==1)])
with open('submission2012.csv', 'w', newline='') as file: 
	filewriter = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	filewriter.writerow(['id', 'PES1'])
	for i, yi in enumerate(y_test2):
		filewriter.writerow([str(i), str(int(yi/2 + 1.5))])

# print running time
stop = time.time()
print('The running time is ', stop - start)



