import pandas as pd
import numpy as np


def attrition_rate(df: pd.DataFrame) -> float:
    return float((df["Attrition"].eq("Yes").mean()) * 100)


def group_attrition(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame()

    out = (
        df.groupby(column, dropna=False)
        .agg(
            Employee_Count=("Attrition", "size"),
            Attrition_Count=("Attrition", lambda s: (s == "Yes").sum()),
            Average_Income=("MonthlyIncome", "mean"),
            Average_Experience=("TotalWorkingYears", "mean"),
        )
        .reset_index()
    )
    out["Attrition_Rate"] = (
        out["Attrition_Count"] / out["Employee_Count"] * 100
    ).round(2)
    return out.sort_values("Attrition_Rate", ascending=False)


def add_dashboard_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Age" in df:
        df["AgeGroup"] = pd.cut(
            df["Age"], [0, 24, 35, 45, 60, np.inf],
            labels=["Under 25", "25-35", "36-45", "46-60", "60+"],
            include_lowest=True
        ).astype(str)

    if "TotalWorkingYears" in df:
        df["ExperienceGroup"] = pd.cut(
            df["TotalWorkingYears"], [-np.inf, 2, 5, 10, 20, np.inf],
            labels=["0-2 years", "3-5 years", "6-10 years", "11-20 years", "20+ years"]
        ).astype(str)

    return df
