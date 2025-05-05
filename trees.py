# This file holds the classes for a basic decision tree, boosting ensemble, and serialization helpers
# Utilized in my proprietary gradient boosted trees algorithm for confidence scoring on bets

import numpy as np
from collections import Counter

# Default Parameters
DEPTH = 6
SAMPLES = 20

# Serialization helpers for TreeNode

def node_to_dict(node):
    if node is None:
        return None
    # Leaf node
    if node.value is not None:
        return {"value": node.value}
    # Decision node
    return {
        "feature_index": node.feature_index,
        "threshold": node.threshold,
        "gain": node.gain,
        "left": node_to_dict(node.left),
        "right": node_to_dict(node.right)
    }


def dict_to_node(d):
    if d is None:
        return None
    # Leaf node
    if "value" in d:
        return TreeNode(value=d["value"])
    # Decision node
    node = TreeNode(
        feature=d["feature_index"],
        threshold=d["threshold"],
        gain=d.get("gain")
    )
    node.left = dict_to_node(d.get("left"))
    node.right = dict_to_node(d.get("right"))
    return node

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        # feature_index: index of the feature to split on
        self.feature_index: int = feature
        # threshold: value at which to split the feature
        self.threshold: float = threshold
        # left / right: child nodes (TreeNode or None)
        self.left = left
        self.right = right
        # gain: information gain achieved by this split
        self.gain = gain
        # value: predicted label or residual for leaf nodes; None for decision nodes
        self.value = value

class DecisionTree:
    def __init__(self, min_samples=SAMPLES, max_depth=DEPTH, reg_lambda = 2, criterion='mse'):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.criterion = criterion  # 'mse' or 'gini'
        self.reg_lambda = reg_lambda
        self.root: TreeNode = None

    def _build_tree(self, X, Y, depth=0):
        n_samples, n_features = np.shape(X)
        if n_samples <= self.min_samples or depth >= self.max_depth:
            leaf_value = np.mean(Y) if self.criterion=='mse' else self._most_common_label(Y)
            return TreeNode(value=leaf_value)
        split = self._find_best_split(X, Y)
        if split['gain'] is None or split['gain'] <= 0:
            if self.criterion=='mse':
                sum_residuals = np.sum(Y)
                leaf_value = sum_residuals / (len(Y) + self.reg_lambda)
            else: self._most_common_label(Y)
            return TreeNode(value=leaf_value)
        left = self._build_tree(split['X_left'], split['Y_left'], depth+1)
        right = self._build_tree(split['X_right'], split['Y_right'], depth+1)
        return TreeNode(
            feature=split['feature_index'],
            threshold=split['threshold'],
            left=left, right=right,
            gain=split['gain']
        )

    def _find_best_split(self, X, Y):
        n_samples, n_features = np.shape(X)
        best_gain = -float('inf')
        best = {'gain': None}
        for j in range(n_features):
            thresholds = set(x[j] for x in X)
            for t in thresholds:
                Xl, Xr, Yl, Yr = self._split(X, Y, j, t)
                if not Xl or not Xr:
                    continue
                gain = self._gain(Y, Yl, Yr)
                if gain > best_gain:
                    best_gain = gain
                    best = {'feature_index': j, 'threshold': t,
                            'X_left': Xl, 'X_right': Xr,
                            'Y_left': Yl, 'Y_right': Yr,
                            'gain': gain}
        return best

    def _split(self, X, Y, idx, thr):
        Xl, Xr, Yl, Yr = [], [], [], []
        for i,x in enumerate(X):
            if x[idx] <= thr:
                Xl.append(x); Yl.append(Y[i])
            else:
                Xr.append(x); Yr.append(Y[i])
        return Xl, Xr, Yl, Yr

    def _gain(self, parent, left, right):
        if self.criterion == 'mse':
            var_p = np.var(parent)
            w_l, w_r = len(left)/len(parent), len(right)/len(parent)
            return var_p - (w_l*np.var(left)+w_r*np.var(right))
        def gini(labels):
            c = Counter(labels)
            return 1 - sum((v/len(labels))**2 for v in c.values())
        gp = gini(parent); w_l = len(left)/len(parent)
        return gp - (w_l*gini(left)+(1-w_l)*gini(right))

    def _most_common_label(self, Y):
        return Counter(Y).most_common(1)[0][0]

    def fit(self, X, Y):
        """Build decision tree on data X, Y."""
        self.root = self._build_tree(X, Y)

    def predict(self, X):
        """Predict values or classes for each sample."""
        return [self._predict_one(x, self.root) for x in X]

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        branch = node.left if x[node.feature_index] <= node.threshold else node.right
        return self._predict_one(x, branch)

    def to_dict(self):
        """Serialize the tree into a JSON-friendly dict."""
        return {
            "min_samples": self.min_samples,
            "max_depth": self.max_depth,
            "criterion": self.criterion,
            "root": node_to_dict(self.root)
        }

    @classmethod
    def from_dict(cls, d):
        """Deserialize a dict into a DecisionTree instance."""
        tree = cls(min_samples=d["min_samples"], max_depth=d["max_depth"], criterion=d["criterion"])
        tree.root = dict_to_node(d["root"])
        return tree

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.05,
                 max_depth=2, min_samples=20, reg_lambda = 2):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []
        self.reg_lambda = reg_lambda
        self.initial_pred = None

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def fit(self, X, y):
        p = np.mean(y)
        self.initial_pred = np.log(p/(1-p))
        F = np.full(len(y), self.initial_pred)
        for m in range(self.n_estimators):
            print('Making New Tree')
            r = y - self._sigmoid(F)
            tree = DecisionTree(min_samples=self.min_samples,
                                max_depth=self.max_depth,
                                reg_lambda=self.reg_lambda,
                                criterion='mse')
            tree.fit(X, r)
            update = np.array(tree.predict(X))
            F += self.lr * update
            self.trees.append(tree)

    def predict_proba(self, X):
        F = np.full(len(X), self.initial_pred)
        for tree in self.trees:
            F += self.lr * np.array(tree.predict(X))
        probs = self._sigmoid(F)
        return np.vstack([1-probs, probs]).T

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:,1] >= threshold).astype(int)

    def to_dict(self):
        return {
            "initial_pred": self.initial_pred,
            "learning_rate": self.lr,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples": self.min_samples,
            "trees": [tree.to_dict() for tree in self.trees]
        }

    @classmethod
    def from_dict(cls, d):
        gbc = cls(n_estimators=d["n_estimators"], learning_rate=d["learning_rate"],
                  max_depth=d["max_depth"], min_samples=d["min_samples"])
        gbc.initial_pred = d["initial_pred"]
        gbc.trees = [DecisionTree.from_dict(td) for td in d["trees"]]
        return gbc
