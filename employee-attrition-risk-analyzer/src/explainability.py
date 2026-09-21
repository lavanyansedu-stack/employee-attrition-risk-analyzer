import pandas as pd


def rule_based_factors(employee: dict):
    """
    Transparent fallback explanation. These are review signals, not proof
    that an employee will leave.
    """
    factors = []

    if employee.get("OverTime") == "Yes":
        factors.append(("Overtime = Yes", "Potential workload signal"))

    if employee.get("JobSatisfaction") in [1, 2]:
        factors.append(("Job Satisfaction is low", "Potential satisfaction signal"))

    if employee.get("WorkLifeBalance") in [1, 2]:
        factors.append(("Work-Life Balance is low", "Potential wellbeing signal"))

    if employee.get("YearsSinceLastPromotion", 0) >= 3:
        factors.append(("Long time since promotion", "Potential career progression signal"))

    if employee.get("YearsAtCompany", 99) <= 2:
        factors.append(("Short tenure", "Potential early-tenure signal"))

    if employee.get("DistanceFromHome", 0) >= 20:
        factors.append(("Long distance from home", "Potential commuting signal"))

    if employee.get("MonthlyIncome", 10**9) < 3000:
        factors.append(("Lower monthly income in this dataset", "Potential compensation-review signal"))

    return factors


def get_model_feature_importance(model, top_n=8):
    """
    Returns global model importance. This is not a causal explanation
    for an individual employee.
    """
    try:
        pre = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        names = pre.get_feature_names_out()

        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            values = abs(estimator.coef_[0])
        else:
            return pd.DataFrame(columns=["Feature", "Importance"])

        result = pd.DataFrame({
            "Feature": names,
            "Importance": values
        }).sort_values("Importance", ascending=False).head(top_n)

        return result
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])


def shap_available():
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False
