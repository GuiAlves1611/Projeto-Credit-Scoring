import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, classification_report

from .build_pipeline import build_pipeline
from .scoring import fit_score_scale, proba_to_score, rating, decision_by_score
from .features_history import build_history_features


# ------------------------------------------------------------
# Helpers (produção realista)
# ------------------------------------------------------------
def _build_scoring_dataset(
    df_clients: pd.DataFrame,
    df_record: pd.DataFrame | None = None,
    window_months: int = 12,
) -> pd.DataFrame:
    """
    Junta cadastro + features derivadas do histórico.
    Se df_record=None, retorna df_clients como está (útil p/ debug).
    """
    df = df_clients.copy()

    if df_record is None:
        return df

    hist = build_history_features(df_record, window_months=window_months)

    df = df.merge(hist, on="ID", how="left")

    # Defaults para quem NÃO tem histórico (muito comum em produção)
    if "n_months" in df.columns:
        df["n_months"] = df["n_months"].fillna(0).astype(int)
    if "vintage" in df.columns:
        df["vintage"] = df["vintage"].fillna(0).astype(int)

# status numérico: preencher ausência de histórico com 0
    for c in ["max_status", "last_status"]:
        if c in df.columns:
            # Garante que seja numérico e preenche NaNs (o seu "X") com 0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df


def _align_to_training_schema(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Garante MESMAS colunas (e ordem) que o modelo viu no treino.
    """
    df = df.copy()
    for c in feature_columns:
        if c not in df.columns:
            df[c] = np.nan
    return df[feature_columns]


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
def train_score_pipeline(
    df,
    target_col="target",
    vintage_col="vintage",
    vintage_quantile=0.7,
    threshold=0.55,
    cat_cols=None,
    drop_cols_model=None,
    # âncoras do score
    p_cut=0.90, s_cut=350,
    p_good=0.05, s_good=850,
    score_clip=(300, 850),
):
    """
    Treina o pipeline, avalia, e gera outputs de score no conjunto de teste.

    Retorna:
      - pipeline
      - df_new (teste com score)
      - metrics
      - score_params
      - feature_columns (schema do treino)
    """

    cat_cols = cat_cols or []
    drop_cols_model = drop_cols_model or []

    # 1) split temporal por vintage (fora do pipeline)
    cut = df[vintage_col].quantile(vintage_quantile)

    train = df[df[vintage_col] <= cut].copy()
    test  = df[df[vintage_col] >  cut].copy()

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col].astype(int)

    X_test = test.drop(columns=[target_col])
    y_test = test[target_col].astype(int)

    # 👇 congela schema que o modelo viu no treino
    feature_columns = X_train.columns.tolist()

    # 2) build pipeline
    pipeline = build_pipeline(cat_cols=cat_cols, drop_cols_model=drop_cols_model)

    # 3) Fit
    pipeline.fit(X_train, y_train)

    # 4) Avaliação (train/test)
    proba_train = pipeline.predict_proba(X_train)[:, 1]
    pred_train = (proba_train >= threshold).astype(int)
    auc_train = roc_auc_score(y_train, proba_train)

    proba_test = pipeline.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= threshold).astype(int)
    auc_test = roc_auc_score(y_test, proba_test)

    metrics = {
        "threshold": float(threshold),
        "auc_train": float(auc_train),
        "auc_test": float(auc_test),
        "gap_auc": float(auc_train - auc_test),
        "report_train": classification_report(y_train, pred_train, zero_division=0),
        "report_test": classification_report(y_test, pred_test, zero_division=0),
    }

    # 5) Score (A/B) e outputs de negócio (no TEST)
    A, B = fit_score_scale(p_cut, s_cut, p_good, s_good)

    df_new = X_test.copy()
    df_new["y_true"] = y_test.values
    df_new["proba_bad"] = proba_test
    df_new["score"] = proba_to_score(
        df_new["proba_bad"], A, B,
        clip_min=score_clip[0], clip_max=score_clip[1]
    )
    
    score_cuts = {
        "q90": 750, "q70": 650, "q40": 570, "q15": 450,
        "cut_reprovado": 450, "cut_manual": 570, "cut_restricao": 650,
    }
    df_new["rating"] = df_new["score"].apply(lambda s: rating(s, score_cuts))
    df_new["decision"] = df_new["score"].apply(lambda s: decision_by_score(s, score_cuts))

    score_params = {
        "A": float(A), "B": float(B),
        "p_cut": float(p_cut), "s_cut": float(s_cut),
        "p_good": float(p_good), "s_good": float(s_good),
        "score_clip_min": float(score_clip[0]),
        "score_clip_max": float(score_clip[1]),
        "score_cuts": score_cuts
    }

    return pipeline, df_new, metrics, score_params, feature_columns


# ------------------------------------------------------------
# Apply (jeito antigo - só funciona se df_new já vier completo)
# ------------------------------------------------------------
def apply_pipeline_to_new_data(df_new, pipeline, score_params, score_clip=(300, 850)):
    """
    df_new: DataFrame só com features (sem target).
    (⚠️ Pressupõe que df_new já tenha as features do histórico.)
    """
    df_new = df_new.copy()
    proba = pipeline.predict_proba(df_new)[:, 1]
    A, B = score_params["A"], score_params["B"]
    cuts = score_params["score_cuts"]

    df_new["proba_bad"] = proba
    df_new["score"] = proba_to_score(
        df_new["proba_bad"], A, B,
        clip_min=score_clip[0], clip_max=score_clip[1]
    )
    df_new["rating"] = df_new["score"].apply(lambda s: rating(s, cuts))
    df_new["decision"] = df_new["score"].apply(lambda s: decision_by_score(s, cuts))

    return df_new


# ------------------------------------------------------------
# Apply (produção realista - cadastro + histórico)
# ------------------------------------------------------------
def apply_pipeline_with_history(
    df_clients_new: pd.DataFrame,
    df_record_new: pd.DataFrame,
    pipeline,
    score_params,
    feature_columns: list[str],
    window_months: int = 12,
    score_clip=(300, 850),
):
    """
    Produção realista:
      - chega df_clients_new (cadastro)
      - chega df_record_new (histórico)
      - gera history_features, merge, alinha schema e aplica pipeline
    """
    df_scoring = _build_scoring_dataset(df_clients_new, df_record_new, window_months=window_months)

    # alinha colunas e ordem igual ao treino
    X = _align_to_training_schema(df_scoring, feature_columns)

    cuts = score_params["score_cuts"]
    proba = pipeline.predict_proba(X)[:, 1]
    A, B = score_params["A"], score_params["B"]

    df_scoring["proba_bad"] = proba
    df_scoring["score"] = proba_to_score(
        df_scoring["proba_bad"], A, B,
        clip_min=score_clip[0], clip_max=score_clip[1]
    )
    df_scoring["rating"] = df_scoring["score"].apply(lambda s: rating(s, cuts))
    df_scoring["decision"] = df_scoring["score"].apply(lambda s: decision_by_score(s, cuts))

    return df_scoring
