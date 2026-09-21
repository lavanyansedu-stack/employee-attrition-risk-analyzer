import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Risk Analyzer",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #eef2ff 0%,
        #f8fafc 100%
    );
    border: 1px solid #e2e8f0;
    margin-bottom: 25px;
}

.hero h1 {
    color: #1e293b;
    margin-bottom: 8px;
}

.hero p {
    color: #64748b;
    font-size: 17px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.risk-high {
    padding: 20px;
    border-radius: 15px;
    background-color: #fee2e2;
    border: 1px solid #fecaca;
    color: #991b1b;
}

.risk-medium {
    padding: 20px;
    border-radius: 15px;
    background-color: #fef3c7;
    border: 1px solid #fde68a;
    color: #92400e;
}

.risk-low {
    padding: 20px;
    border-radius: 15px;
    background-color: #dcfce7;
    border: 1px solid #bbf7d0;
    color: #166534;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FIND DATASET
# ============================================================

def find_dataset():

    current_dir = Path(__file__).resolve().parent
    project_dir = current_dir.parent

    possible_files = [

        # app folder
        current_dir / "employee_attrition.csv",
        current_dir / "employee_attrition.csv.csv",

        # project folder
        project_dir / "employee_attrition.csv",
        project_dir / "employee_attrition.csv.csv",

        # data folder
        project_dir / "data" / "employee_attrition.csv",
        project_dir / "data" / "employee_attrition.csv.csv",

        # common IBM HR dataset name
        project_dir / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
        project_dir / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    ]

    for file in possible_files:
        if file.exists():
            return file

    # Search recursively
    for pattern in [
        "*.csv",
        "*.CSV"
    ]:

        files = list(project_dir.rglob(pattern))

        for file in files:

            name = file.name.lower()

            if (
                "attrition" in name
                or "employee" in name
                or "hr" in name
            ):
                return file

    return None


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    dataset_path = find_dataset()

    if dataset_path is None:
        return None, None

    try:
        df = pd.read_csv(dataset_path)

        return df, dataset_path

    except Exception as e:

        st.error(f"Error loading dataset: {e}")

        return None, dataset_path


df, dataset_path = load_data()


# ============================================================
# CHECK DATASET
# ============================================================

if df is None:

    st.error("❌ Employee Attrition dataset was not found.")

    st.markdown("""
    ### Expected dataset

    Place your CSV file in one of these locations:

    ```text
    employee-attrition-risk-analyzer/
    ├── employee_attrition.csv
    ├── employee_attrition.csv.csv
    │
    └── data/
        └── employee_attrition.csv
    ```

    The application will automatically search for the dataset.
    """)

    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# CHECK TARGET COLUMN
# ============================================================

if "Attrition" not in df.columns:

    st.error(
        "❌ The dataset does not contain an 'Attrition' column."
    )

    st.write("Available columns:")

    st.write(list(df.columns))

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

data = df.copy()

# Remove duplicate rows
data = data.drop_duplicates()

# Remove constant columns
constant_columns = [
    column
    for column in data.columns
    if data[column].nunique(dropna=False) <= 1
]

data = data.drop(columns=constant_columns, errors="ignore")


# ============================================================
# TARGET ENCODING
# ============================================================

data["Attrition"] = (
    data["Attrition"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0
    })
)

data = data.dropna(subset=["Attrition"])

data["Attrition"] = data["Attrition"].astype(int)


# ============================================================
# REMOVE IDENTIFIER COLUMNS
# ============================================================

identifier_columns = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

identifier_columns = [
    col for col in identifier_columns
    if col in data.columns
]

data = data.drop(
    columns=identifier_columns,
    errors="ignore"
)


# ============================================================
# CREATE FEATURES AND TARGET
# ============================================================

X = data.drop(columns=["Attrition"])

y = data["Attrition"]


# ============================================================
# IDENTIFY COLUMN TYPES
# ============================================================

numeric_columns = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(X_train, y_train):

    trained_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]
    )

    trained_model.fit(
        X_train,
        y_train
    )

    return trained_model


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Train model
model = train_model(
    X_train,
    y_train
)


# ============================================================
# MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

try:

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

except Exception:

    roc_auc = 0


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔎 Employee Analyzer")

st.sidebar.markdown(
    "AI-powered employee attrition risk analysis"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Analyze Employee",
        "Dataset",
        "Model Performance"
    ]
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

<h1>🔎 Employee Attrition Risk Analyzer</h1>

<p>
AI-powered HR decision-support dashboard for analyzing
employee attrition patterns and estimating employee risk.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.subheader("📊 HR Dashboard")

    total_employees = len(data)

    employees_left = int(
        data["Attrition"].sum()
    )

    employees_stayed = (
        total_employees - employees_left
    )

    attrition_rate = (
        employees_left /
        total_employees *
        100
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Employees",
            total_employees
        )

    with col2:

        st.metric(
            "Employees Left",
            employees_left
        )

    with col3:

        st.metric(
            "Employees Stayed",
            employees_stayed
        )

    with col4:

        st.metric(
            "Attrition Rate",
            f"{attrition_rate:.1f}%"
        )


    st.divider()

    # --------------------------------------------------------
    # ATTRITION DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Employee Attrition Distribution"
        )

        attrition_counts = (
            data["Attrition"]
            .map({
                0: "Stayed",
                1: "Left"
            })
            .value_counts()
        )

        st.bar_chart(
            attrition_counts
        )


    with col2:

        st.subheader(
            "Model Performance"
        )

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )


    # --------------------------------------------------------
    # OVERTIME ANALYSIS
    # --------------------------------------------------------

    if "OverTime" in data.columns:

        st.divider()

        st.subheader(
            "⏱️ Attrition by Overtime"
        )

        overtime_table = (
            data.groupby("OverTime")["Attrition"]
            .mean()
            .mul(100)
            .round(2)
        )

        st.bar_chart(
            overtime_table
        )


    # --------------------------------------------------------
    # JOB ROLE ANALYSIS
    # --------------------------------------------------------

    if "JobRole" in data.columns:

        st.subheader(
            "💼 Attrition by Job Role"
        )

        job_role_table = (
            data.groupby("JobRole")["Attrition"]
            .mean()
            .mul(100)
            .sort_values(
                ascending=False
            )
            .round(2)
        )

        st.bar_chart(
            job_role_table
        )


# ============================================================
# ANALYZE EMPLOYEE
# ============================================================

elif page == "Analyze Employee":

    st.subheader(
        "🔎 Analyze Employee"
    )

    st.write(
        "Enter employee information to estimate "
        "attrition risk."
    )

    input_data = {}

    # --------------------------------------------------------
    # CREATE INPUT FORM
    # --------------------------------------------------------

    with st.form("employee_form"):

        st.markdown(
            "### 👤 Employee Information"
        )

        columns = st.columns(3)

        for index, column in enumerate(X.columns):

            with columns[index % 3]:

                # Numeric input
                if column in numeric_columns:

                    minimum = float(
                        X[column].min()
                    )

                    maximum = float(
                        X[column].max()
                    )

                    median = float(
                        X[column].median()
                    )

                    # Integer columns
                    if pd.api.types.is_integer_dtype(
                        X[column]
                    ):

                        input_data[column] = st.number_input(
                            column,
                            min_value=int(minimum),
                            max_value=int(maximum),
                            value=int(median),
                            step=1
                        )

                    else:

                        input_data[column] = st.number_input(
                            column,
                            min_value=float(minimum),
                            max_value=float(maximum),
                            value=float(median)
                        )

                # Categorical input
                else:

                    values = (
                        X[column]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    values = sorted(values)

                    if len(values) > 0:

                        input_data[column] = st.selectbox(
                            column,
                            values
                        )

        submitted = st.form_submit_button(
            "🔍 ANALYZE EMPLOYEE",
            use_container_width=True
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if submitted:

        feature_df = pd.DataFrame(
            [input_data]
        )

        try:

            prediction = model.predict(
                feature_df
            )[0]

            probabilities = model.predict_proba(
                feature_df
            )

            # Find probability for class 1 safely
            classifier = model.named_steps[
                "classifier"
            ]

            classes = list(
                classifier.classes_
            )

            if 1 in classes:

                positive_index = classes.index(1)

                risk_probability = (
                    probabilities[0][positive_index]
                )

            else:

                risk_probability = 0.0


            risk_percentage = (
                risk_probability * 100
            )


            # ------------------------------------------------
            # RISK LEVEL
            # ------------------------------------------------

            if risk_percentage >= 60:

                risk_level = "HIGH"

                st.markdown(
                    f"""
                    <div class="risk-high">

                    <h2>🔴 HIGH ATTRITION RISK</h2>

                    <h3>
                    Risk Score: {risk_percentage:.1f}%
                    </h3>

                    <p>
                    The model estimates a relatively high
                    probability of attrition for this input.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif risk_percentage >= 30:

                risk_level = "MEDIUM"

                st.markdown(
                    f"""
                    <div class="risk-medium">

                    <h2>🟡 MEDIUM ATTRITION RISK</h2>

                    <h3>
                    Risk Score: {risk_percentage:.1f}%
                    </h3>

                    <p>
                    The model estimates a moderate
                    probability of attrition for this input.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                risk_level = "LOW"

                st.markdown(
                    f"""
                    <div class="risk-low">

                    <h2>🟢 LOW ATTRITION RISK</h2>

                    <h3>
                    Risk Score: {risk_percentage:.1f}%
                    </h3>

                    <p>
                    The model estimates a relatively low
                    probability of attrition for this input.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            st.divider()

            # ------------------------------------------------
            # PREDICTION RESULT
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Risk Score",
                    f"{risk_percentage:.1f}%"
                )

            with col2:

                st.metric(
                    "Risk Level",
                    risk_level
                )

            with col3:

                prediction_text = (
                    "Potential Attrition"
                    if prediction == 1
                    else "Likely Retention"
                )

                st.metric(
                    "Prediction",
                    prediction_text
                )


            # ------------------------------------------------
            # EMPLOYEE FACTORS
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "📋 Employee Information"
            )

            display_df = pd.DataFrame(
                {
                    "Feature": list(
                        input_data.keys()
                    ),
                    "Value": list(
                        input_data.values()
                    )
                }
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


            st.info(
                "⚠️ This risk score is a machine-learning "
                "estimate based on historical dataset patterns. "
                "It should not be used as the sole basis for "
                "employment decisions."
            )


        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "Dataset":

    st.subheader(
        "📁 Employee Dataset"
    )

    st.write(
        f"Dataset: `{dataset_path.name}`"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            data.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            data.shape[1]
        )

    with col3:

        st.metric(
            "Missing Values",
            int(data.isnull().sum().sum())
        )


    st.divider()

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        data.head(20),
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "Dataset Statistics"
    )

    st.dataframe(
        data.describe(
            include="all"
        ).transpose(),
        use_container_width=True
    )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":

    st.subheader(
        "🤖 Machine Learning Model Performance"
    )

    st.write(
        "Model used: Logistic Regression"
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

    with col5:

        st.metric(
            "ROC-AUC",
            f"{roc_auc:.3f}"
        )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Stay",
            "Actual Leave"
        ],
        columns=[
            "Predicted Stay",
            "Predicted Leave"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )


    # --------------------------------------------------------
    # METRICS TABLE
    # --------------------------------------------------------

    st.subheader(
        "Evaluation Summary"
    )

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],
            "Score": [
                f"{accuracy:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                f"{roc_auc:.4f}"
            ]
        }
    )

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Employee Attrition Risk Analyzer | "
    "Python • Pandas • Scikit-learn • Streamlit"
)

st.caption(
    "Prototype HR decision-support system — "
    "model predictions should be reviewed by qualified humans."
)