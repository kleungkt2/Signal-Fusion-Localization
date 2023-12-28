from itertools import combinations_with_replacement
import numpy as np
from sklearn.svm import OneClassSVM
import pickle
import os


class InOutClassifier():
    def __init__(self):
        self.base_learners = {}
        self.n_features = -1

    def train(self, X):
        self.n_features = X.shape[1]
        for feature_indices in combinations_with_replacement(range(self.n_features), 2):
            clf = OneClassSVM(nu=0.1, gamma=0.005)
            tmp = X[:, feature_indices]
            clf.fit(tmp[np.sum(tmp, axis=1) != 0])
            self.base_learners[feature_indices] = clf

    def predict(self, features):
        on_features = np.flatnonzero(features)
        sums = [0 for _ in range(self.n_features)]
        weights = [0 for _ in range(self.n_features)]

        for feature_indices in combinations_with_replacement(on_features, 2):
            output = self.base_learners[feature_indices].predict(features[list(feature_indices)].reshape(1, -1))[0]
            # print("{}: {}".format(feature_indices, output))
            for i in range(2):
                weight = features[feature_indices[1-i]]
                sums[feature_indices[i]] += weight * output
                weights[feature_indices[i]] += weight

                if feature_indices[0] == feature_indices[1]:
                    break

        sums = [s/w if w != 0 else 0 for s, w in zip(sums, weights)]

        return np.average(sums, weights=features)

    def save(self, save_dir):
        with open(os.path.join(save_dir, 'inout'), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    """Use Libsvm for integration to C++ code"""
    # def train(self, X):
    #     self.n_features = X.shape[1]
    #     # i = 0
    #     for feature_indices in combinations_with_replacement(range(self.n_features), 2):
    #         tmp = X[:, feature_indices]
    #         tmp = tmp[np.sum(tmp, axis=1) != 0]
    #         clf = svm_train(np.zeros(len(tmp)), tmp, '-s 2 -t 2 -g 0.005 -n 0.1 -q')
    #         self.base_learners[feature_indices] = clf
    #         # print(i)
    #         # i+=1

    # def predict(self, features):
    #     on_features = np.flatnonzero(features)
    #     sums = [0 for _ in range(self.n_features)]
    #     weights = [0 for _ in range(self.n_features)]
    #     # indices = dict(zip(on_features,[np.where(on_features==x)[0][0] for x in on_features]))

    #     for feature_indices in combinations_with_replacement(on_features, 2):
    #         clf = self.base_learners[feature_indices]
    #         output, _, _ = svm_predict([0], features[list(feature_indices)].reshape(1,-1), clf, '-q')
    #         # print("{}: {}".format(feature_indices, output))
    #         for i in range(2):
    #             weight = features[feature_indices[1-i]]
    #             sums[feature_indices[i]] += weight * output[0]
    #             weights[feature_indices[i]] += weight

    #             if feature_indices[0] == feature_indices[1]:
    #                 break

    #     sums = [s/w if w != 0 else 0 for s,w in zip(sums, weights)]

    #     return np.average(sums, weights=features)

