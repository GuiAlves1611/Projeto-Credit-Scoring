import pandas as pd

from .features_history import build_history_features

def build_scoring_df(df_clients_new: pd.DataFrame, hist_features: pd.DataFrame) -> pd.DataFrame:
    """
    Monta o dataset de scoring (produção):
    - merge cadastro + features do histórico (já pré-calculadas)
    - aplica defaults
    - garante preenchimento dos status NUMÉRICOS
    """
    df_new = df_clients_new.merge(hist_features, on="ID", how="left")

    # defaults (igual produção real)
    if "n_months" in df_new.columns:
        df_new["n_months"] = df_new["n_months"].fillna(0).astype(int)

    if "vintage" in df_new.columns:
        df_new["vintage"] = df_new["vintage"].fillna(0).astype(int)

    # ✅ status NUMÉRICOS (novo)
    for c in ["max_status_num", "last_status_num"]:
        if c in df_new.columns:
            df_new[c] = df_new[c].fillna(0).astype(int)

    return df_new

def _build_scoring_dataset(
    df_clients: pd.DataFrame,
    df_record: pd.DataFrame | None = None,
    window_months: int = 12,
) -> pd.DataFrame:
    """
    Junta cadastro + features derivadas do histórico.
    """
    df = df_clients.copy()

    if df_record is None:
        return df

    # ⚠️ você já tem isso no seu projeto:
    # from .features_history import build_history_features
    hist = build_history_features(df_record, window_months=window_months)

    df = df.merge(hist, on="ID", how="left")

    # Defaults para quem NÃO tem histórico
    if "n_months" in df.columns:
        df["n_months"] = df["n_months"].fillna(0).astype(int)

    if "vintage" in df.columns:
        df["vintage"] = df["vintage"].fillna(0).astype(int)

    # ✅ status NUMÉRICOS (novo)
    for c in ["max_status", "last_status", "n_months", "vintage", "last_month"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df

def prepare_X_for_model(df_new: pd.DataFrame):
    """
    Prepara X para o modelo e preserva o ID.

    Retorna:
        X (DataFrame): features para o pipeline
        ids (Series): IDs dos clientes
    """
    df_new_model = df_new.copy()

    ids = None
    if "ID" in df_new_model.columns:
        ids = df_new_model["ID"].copy()
        df_new_model = df_new_model.drop(columns=["ID"])

    CAT_COLS = [
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
    ]

    NUM_COLS = ["max_status", "last_status", "n_months", "vintage", "last_month"]
    for c in NUM_COLS:
        if c in df_new_model.columns:
            df_new_model[c] = pd.to_numeric(df_new_model[c], errors="coerce").fillna(0)

    for c in CAT_COLS:
        if c in df_new_model.columns:
            df_new_model[c] = df_new_model[c].astype("category")

    return df_new_model, ids

