
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

#Transformers / Estimator

class DropCols(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = [c for c in self.cols_to_drop if c in X.columns]
        return X.drop(columns=cols, errors="ignore")

class EnsureNumeric(BaseEstimator, TransformerMixin):
    """
    Garante que colunas específicas estejam numéricas (float/int).
    Isso evita XGBoostError por dtype errado no apply.
    """
    def __init__(self, num_cols=None, fillna_value=0):
        self.num_cols = num_cols or []
        self.fillna_value = fillna_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.num_cols:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(self.fillna_value)
        return X

class LogTransform(BaseEstimator, TransformerMixin):
    """
    Aplica log1p em colunas numéricas.
    """
    def __init__(self, cols=None):
        self.cols = cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            if c in X.columns:
                X[c] = np.log1p(X[c].clip(lower=0))
        return X


class EnsureCategorical(BaseEstimator, TransformerMixin):
    """
    Garante que colunas categóricas do pandas estejam em dtype 'category'
    (importante para XGBoost com enable_categorical=True).
    """
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
        return X


class XGBWithAutoSPW(BaseEstimator, ClassifierMixin):
    """
    Wrapper para XGBClassifier calculando scale_pos_weight automaticamente no fit
    (neg/pos) caso não seja fornecido.
    """
    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.model_ = None
        self.scale_pos_weight_ = None

    def fit(self, X, y):
        y = np.asarray(y)
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        self.scale_pos_weight_ = (neg / pos) if pos > 0 else 1.0

        params = dict(self.xgb_params)
        params.setdefault("scale_pos_weight", self.scale_pos_weight_)

        self.model_ = xgb.XGBClassifier(**params)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
