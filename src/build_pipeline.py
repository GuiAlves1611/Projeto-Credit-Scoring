from sklearn.pipeline import Pipeline

from .pipeline_components import (
    EnsureCategorical,
    DropCols,
    XGBWithAutoSPW
)

def build_pipeline(cat_cols, drop_cols_model):
    """
    Constrói o pipeline padrão do projeto de crédito.
    Retorna um sklearn Pipeline pronto para fit/predict.
    """

    pipeline = Pipeline([
        ("ensure_cat", EnsureCategorical(cat_cols=list(cat_cols))),

        ("drop", DropCols(cols_to_drop=list(drop_cols_model))),

        ("model", XGBWithAutoSPW(
            objective="binary:logistic",
            enable_categorical=True,

            # Parâmetros ajustados via tuning
            subsample=0.9,
            reg_lambda=5,
            reg_alpha=0.3,
            n_estimators=600,
            min_child_weight=20,
            max_depth=2,
            learning_rate=0.01,
            colsample_bytree=0.8,
            gamma=0,
        ))
    ])
    return pipeline