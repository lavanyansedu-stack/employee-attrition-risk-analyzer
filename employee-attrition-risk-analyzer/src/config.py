from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "employee_attrition.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "attrition_model.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"
FEATURE_INFO_PATH = PROJECT_ROOT / "models" / "feature_info.json"
CHART_DIR = PROJECT_ROOT / "reports" / "charts"

RISK_LOW = 0.30
RISK_MEDIUM = 0.60

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Columns that are constant, identifiers, or otherwise unsuitable for learning.
DROP_COLUMNS = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours",
]

TARGET = "Attrition"
