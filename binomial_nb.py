import re
import csv
import string
from collections import Counter

import numpy as np
import pandas as pd


class BernoulliNB:

    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_, y_numeric, cls_counts  = np.unique(y, return_inverse=True, return_counts=True)
        self.n_classes_ = len(self.classes_)

        self.cls_prior_ = cls_counts / len(y)
        self._feat_prob_ = np.zeros((self.n_classes_, X.shape[1]))
        for idx in range(self.n_classes_):
            self._feat_prob_[idx] = X[np.argwhere(y_numeric==idx)].sum(axis=0)
        self._feat_prob_ += self.alpha

        class_sums = cls_counts + (self.n_classes_ * self.alpha)
        self._feat_prob_ = self._feat_prob_ / class_sums[:, np.newaxis]

    def predict_log_proba(self, X):
        neg_prob = np.log(1 - self._feat_prob_)
        return np.dot(X, (np.log(self._feat_prob_) - neg_prob).T) \
                + np.log(self.cls_prior_) \
                + neg_prob.sum(axis=1)

    def predict(self, X):
        _, n_features = self._feat_prob_.shape
        n_features_X = X.shape[1]

        if n_features_X != n_features:
            raise ValueError("Expected input with %d features, got %d instead"
                             % (n_features, n_features_X))
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]
