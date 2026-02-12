import pandas as pd

BAD = {2, 3, 4, 5}
STATUS_MAP = {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"C":0,"X":0}

def build_history_features(df_record: pd.DataFrame, window_months: int = 12) -> pd.DataFrame:
    cr = df_record.copy()
    cr["STATUS"] = cr["STATUS"].astype(str)
    cr["MONTHS_BALANCE"] = cr["MONTHS_BALANCE"].astype(int)

    w = cr[(cr["MONTHS_BALANCE"] <= 0) & (cr["MONTHS_BALANCE"] >= -window_months)].copy()
    w = w.sort_values(["ID", "MONTHS_BALANCE"], ascending=[True, False])

    # ✅ transforma STATUS em severidade numérica
    w["STATUS"] = w["STATUS"].map(STATUS_MAP).fillna(0).astype(int)

    agg = (
        w.groupby("ID")
         .agg(
            max_status=("STATUS", "max"),
            last_status=("STATUS", "first"),
            n_months=("MONTHS_BALANCE", "count"),
            last_month=("MONTHS_BALANCE", "max"),
         )
         .reset_index()
    )

    tmp_bad = w[w["STATUS"].isin(BAD)].copy()
    last_bad = (
        tmp_bad.groupby("ID")["MONTHS_BALANCE"]
            .max()
            .rename("last_bad")
            .reset_index()
    )


    vintage = (
        w.groupby("ID")["MONTHS_BALANCE"]
         .min().abs()
         .rename("vintage")
         .reset_index()
    )

    out = (
        agg
        .merge(vintage, on="ID", how="left")
        .merge(last_bad, on="ID", how="left")
    )
    out["last_bad"] = out["last_bad"].fillna(-1)
    return out
