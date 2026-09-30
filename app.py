import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="AttritionAI", page_icon="👥", layout="wide")
ROOT = Path(__file__).parent
DEMO = ROOT/"data"/"employee_attrition_demo.csv"
OFFICIAL = ROOT/"data"/"employee_attrition.csv"

TARGET="Attrition"
DROP=["Attrition","EmployeeNumber","EmployeeCount","StandardHours","Over18"]

@st.cache_data
def read_csv(path):
    return pd.read_csv(path)

def train_all(df):
    # Target column is Attrition.
    # AttritionFlag must NEVER be used as a model feature.
    X = df.drop(
        columns=[
            c for c in DROP + ["AttritionFlag"]
            if c in df.columns
        ]
    )

    y = (
        df[TARGET]
        .astype(str)
        .str.lower()
        .eq("yes")
        .astype(int)
    )

    cats = X.select_dtypes(include="object").columns.tolist()
    nums = [c for c in X.columns if c not in cats]

    prep = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]),
            nums
        ),
        (
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cats
        )
    ])

    models = {
        "Logistic Regression":
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced"
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=250,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=2
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=7,
                random_state=42,
                class_weight="balanced"
            )
    }

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    fitted = {}
    rows = []

    for name, model in models.items():

        # Create a fresh preprocessing pipeline for every model
        model_prep = ColumnTransformer([
            (
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())
                ]),
                nums
            ),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                cats
            )
        ])

        pipe = Pipeline([
            ("prep", model_prep),
            ("model", model)
        ])

        pipe.fit(Xtr, ytr)

        pr = pipe.predict(Xte)
        pb = pipe.predict_proba(Xte)[:, 1]

        fitted[name] = pipe

        rows.append([
            name,
            accuracy_score(yte, pr),
            precision_score(yte, pr, zero_division=0),
            recall_score(yte, pr, zero_division=0),
            f1_score(yte, pr, zero_division=0),
            roc_auc_score(yte, pb)
        ])

    metrics = pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ]
    )

    return metrics, fitted, Xte, yte

def band(p):
    return "LOW" if p<.33 else ("MEDIUM" if p<.66 else "HIGH")

def warning_signals(r):
    out=[]
    if r.get("OverTime")=="Yes": out.append("Overtime workload")
    if float(r.get("JobSatisfaction",4))<=2: out.append("Low job satisfaction")
    if float(r.get("EnvironmentSatisfaction",4))<=2: out.append("Low environment satisfaction")
    if float(r.get("WorkLifeBalance",4))<=2: out.append("Work-life balance concern")
    if float(r.get("DistanceFromHome",0))>=20: out.append("Long commute")
    if float(r.get("YearsSinceLastPromotion",0))>=7: out.append("Long promotion gap")
    if float(r.get("NumCompaniesWorked",0))>=5: out.append("High prior job mobility")
    if r.get("BusinessTravel")=="Travel_Frequently": out.append("Frequent business travel")
    return out

st.markdown("""
<style>
.block-container{padding-top:1.1rem}
.hero{padding:1.35rem 1.5rem;border-radius:20px;background:linear-gradient(135deg,#0f172a,#1e3a5f);color:white;margin-bottom:1rem}
.hero h1{margin:0;font-size:2.2rem}.hero p{margin:.4rem 0 0;color:#dbeafe}
.kpi{padding:1rem;border:1px solid #e2e8f0;border-radius:16px;background:white}
.small{color:#64748b;font-size:.82rem}
</style>
""",unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>👥 AttritionAI</h1>
<p>Employee Attrition Analysis & Prediction • original hackathon prototype</p></div>""",unsafe_allow_html=True)

with st.sidebar:
    st.header("Data")
    up=st.file_uploader("Upload CSV",type="csv")
    if up:
        data=pd.read_csv(up); source="Uploaded dataset"
    elif OFFICIAL.exists():
        data=read_csv(OFFICIAL); source="Organizer dataset"
    else:
        data=read_csv(DEMO); source="Bundled demo dataset"
    st.caption(f"Source: {source}")
    page=st.radio("Module",["Dashboard","Attrition Analysis","Model Lab","Risk Predictor","Data Explorer"])
    st.divider()
    st.warning("Decision-support only: do not use a prediction as the sole basis for employment decisions.")

if TARGET not in data.columns:
    st.error("CSV must contain an 'Attrition' column with Yes/No values.")
    st.stop()

data=data.copy()
data["AttritionFlag"]=data[TARGET].astype(str).str.lower().eq("yes").astype(int)
metrics,models,Xte,yte=train_all(data)
rf=models["Random Forest"]

if page=="Dashboard":
    c1,c2,c3,c4=st.columns(4)
    vals=[
        ("Employees",f"{len(data):,}"),
        ("Attrition Rate",f"{data.AttritionFlag.mean()*100:.1f}%"),
        ("Departments",str(data["Department"].nunique()) if "Department" in data else "-"),
        ("Job Roles",str(data["JobRole"].nunique()) if "JobRole" in data else "-")
    ]
    for c,(a,b) in zip([c1,c2,c3,c4],vals):
        c.markdown(f'<div class="kpi"><div class="small">{a}</div><h2>{b}</h2></div>',unsafe_allow_html=True)
    st.subheader("Workforce overview")
    if "OverTime" in data:
        st.bar_chart(pd.crosstab(data["OverTime"],data["Attrition"],normalize="index")["Yes"]*100)
    st.info("Charts show dataset associations; they do not establish that one factor causes attrition.")

elif page=="Attrition Analysis":
    st.subheader("🔎 Attrition Pattern Analysis")
    a,b=st.columns(2)
    with a:
        if "JobRole" in data:
            role=pd.crosstab(data["JobRole"],data["Attrition"],normalize="index")["Yes"].sort_values(ascending=False)*100
            st.bar_chart(role)
            st.caption("Attrition rate by job role (%)")
    with b:
        if "Department" in data:
            dept=pd.crosstab(data["Department"],data["Attrition"],normalize="index")["Yes"]*100
            st.bar_chart(dept)
            st.caption("Attrition rate by department (%)")
    st.subheader("Potential signals")
    st.write("Overtime • satisfaction • work-life balance • commute • promotion gap • travel • job level")

elif page=="Model Lab":
    st.subheader("🧪 Model Lab")
    st.dataframe(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c!="Model"}),use_container_width=True)
    st.caption("Metrics use a stratified 80/20 hold-out split.")
    pr=rf.predict(Xte)
    st.subheader("Random Forest confusion matrix")
    cm=confusion_matrix(yte,pr)
    st.dataframe(pd.DataFrame(cm,index=["Actual Stay","Actual Leave"],columns=["Predicted Stay","Predicted Leave"]))

elif page=="Risk Predictor":
    st.subheader("🎯 Individual Risk Predictor")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 18, 65, 30)
        income = st.number_input("Monthly Income", 1000, 30000, 5000, step=500)
        distance = st.number_input("Distance From Home", 1, 60, 8)
        years = st.number_input("Years At Company", 0, 45, 4)
        level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)

    with c2:
        overtime = st.selectbox("Overtime", ["No", "Yes"])
        js = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
        es = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2)
        wb = st.selectbox("Work-Life Balance", [1, 2, 3, 4], index=2)
        ji = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2)

    with c3:
        travel = st.selectbox(
            "Business Travel",
            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
            index=1
        )

        marital = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"],
            index=1
        )

        role = st.selectbox(
            "Job Role",
            [
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources"
            ]
        )

        companies = st.number_input("Previous Companies", 0, 10, 2)
        promotion = st.number_input("Years Since Last Promotion", 0, 20, 2)

    row = {
        "Age": age,
        "BusinessTravel": travel,
        "DailyRate": 800,
        "Department": "Research & Development",
        "DistanceFromHome": distance,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": es,
        "Gender": "Male",
        "HourlyRate": 65,
        "JobInvolvement": ji,
        "JobLevel": level,
        "JobRole": role,
        "JobSatisfaction": js,
        "MaritalStatus": marital,
        "MonthlyIncome": income,
        "MonthlyRate": 14000,
        "NumCompaniesWorked": companies,
        "OverTime": overtime,
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": max(years + 3, years),
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": wb,
        "YearsAtCompany": years,
        "YearsInCurrentRole": min(3, years),
        "YearsSinceLastPromotion": min(promotion, years),
        "YearsWithCurrManager": min(3, years)
    }

    # Columns used during model training
    features = [
        c for c in data.columns
        if c not in DROP and c != "AttritionFlag"
    ]

    # Create prediction input
    inp = pd.DataFrame([row])

    # Remove target column if it somehow exists
    if "AttritionFlag" in inp.columns:
        inp = inp.drop(columns=["AttritionFlag"])

    # Add any missing training columns
    for col in features:
        if col not in inp.columns:
            if data[col].dtype == "object":
                mode = data[col].mode()
                inp[col] = mode.iloc[0] if not mode.empty else "Unknown"
            else:
                inp[col] = data[col].median()

    # Keep EXACT same columns/order as training data
    inp = inp[features]

    if st.button(
        "🚀 Predict Attrition Risk",
        type="primary",
        use_container_width=True
    ):

        try:
            prob = float(rf.predict_proba(inp)[0, 1])

            st.metric(
                "Estimated attrition probability",
                f"{prob * 100:.1f}%"
            )

            st.subheader(f"Risk band: {band(prob)}")

            flags = warning_signals(row)

            if flags:
                for f in flags:
                    st.warning("• " + f)
            else:
                st.success("No rule-based warning signal triggered.")

            st.info(
                "Use the result to prioritize a human conversation or review. "
                "It is not an automated HR decision."
            )

        except Exception as e:
            st.error("Prediction failed.")
            st.code(str(e))
            st.write("Input columns:", list(inp.columns))
            st.write("Training columns:", features)

else:
    st.subheader("📋 Data Explorer")
    st.dataframe(data.head(100), use_container_width=True)

    st.download_button(
        "Download current CSV",
        data.to_csv(index=False).encode(),
        "employee_attrition.csv",
        "text/csv"
    )