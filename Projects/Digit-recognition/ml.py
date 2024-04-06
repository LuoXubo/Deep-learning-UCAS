"""
@Description :   Mechine Learning for digit recognition
@Author      :   Xubo Luo 
@Time        :   2024/04/06 18:20:02
"""

import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.externals import joblib
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meachine Learning for digit recognition')
    parser.add_argument('--test_size', type=float, default=0.3, metavar='N', help='test size for training (default: 0.3)')
    parser.add_argument('--save_path', type=str, default='./caches/', metavar='S', help='save path for model')
    parser.add_argument('--train', type=bool, default=True, metavar='T', help='whether to train the model (default: True)')
    parser.add_argument('--method', type=str, default='svm', metavar='M', help='which method to use "svm, knn, random_forest" (default: svm)')
    args = parser.parse_args()

    test_size = args.test_size
    save_path = args.save_path
    train_flag = args.train
    method = args.method

    print('--------------------------------------')
    print('test_size: ', test_size)
    print('save_path: ', save_path)
    print('train_flag: ', train_flag)
    print('method: ', method)
    print('--------------------------------------')

    # Load data
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize features
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if method == 'svm':
        clf = svm.SVC(gamma=0.001)
    elif method == 'knn':
        clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    elif method == 'random_forest':
        clf = ensemble.RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError('Invalid method (svm, knn, random_forest)')

    if train_flag:
        clf.fit(X_train, y_train)
        joblib.dump(clf, save_path + method + '.pkl')

    # load model
    clf = joblib.load(save_path + method + '.pkl')
    y_pred = clf.predict(X_test)
    print(method + ' Accuracy: ', metrics.accuracy_score(y_test, y_pred))