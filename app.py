import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="",
    layout="wide"
)

st.title(" Student Performance Predictor")
st.write("Predict a student's final grade using study habits, attendance, academic performance and other factors.")

@st.cache_data
def load_data():
    df = pd.read_csv("student_performance_dataset.csv")
    df["parental_education"] = df["parental_education"].fillna("Unknown")
    return df

df = load_data()

feature_cols = [
    c for c in df.columns
    if c not in ["student_id", "final_grade"]
]

X = df[feature_cols].copy()
y = df["final_grade"].copy()

numeric_features = [
    "study_time_hours",
    "attendance_percent",
    "sleep_hours",
    "previous_grade",
    "final_exam_score"
]

categorical_features = [
    "gender",
    "parental_education",
    "internet_access",
    "extracurricular_activities",
    "part_time_job"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        random_state=42
    )
}

cv_results = {}

for name, model in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=5,
        scoring="f1_macro"
    )

    cv_results[name] = scores.mean()

param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 8, 12],
    "model__min_samples_leaf": [1, 2, 4]
}

rf_pipeline = Pipeline([
    ("prep", preprocessor),
    (
        "model",
        RandomForestClassifier(
            random_state=42
        )
    )
])

grid = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

conf_matrix = confusion_matrix(
    y_test,
    y_pred
)

feature_names = best_model.named_steps[
    "prep"
].get_feature_names_out()

importances = best_model.named_steps[
    "model"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(
    by="Importance",
    ascending=False
)

st.sidebar.header("Student Information")

gender = st.sidebar.selectbox(
    "Gender",
    sorted(df["gender"].dropna().unique())
)

parental_education = st.sidebar.selectbox(
    "Parental Education",
    sorted(df["parental_education"].dropna().unique())
)

internet_access = st.sidebar.selectbox(
    "Internet Access",
    sorted(df["internet_access"].dropna().unique())
)

extracurricular_activities = st.sidebar.selectbox(
    "Extracurricular Activities",
    sorted(df["extracurricular_activities"].dropna().unique())
)

part_time_job = st.sidebar.selectbox(
    "Part Time Job",
    sorted(df["part_time_job"].dropna().unique())
)

study_time_hours = st.sidebar.number_input(
    "Study Time Hours",
    min_value=float(df["study_time_hours"].min()),
    max_value=float(df["study_time_hours"].max()),
    value=float(df["study_time_hours"].median()),
    step=0.5
)

attendance_percent = st.sidebar.number_input(
    "Attendance Percentage",
    min_value=float(df["attendance_percent"].min()),
    max_value=float(df["attendance_percent"].max()),
    value=float(df["attendance_percent"].median()),
    step=1.0
)

sleep_hours = st.sidebar.number_input(
    "Sleep Hours",
    min_value=float(df["sleep_hours"].min()),
    max_value=float(df["sleep_hours"].max()),
    value=float(df["sleep_hours"].median()),
    step=0.5
)

previous_grade = st.sidebar.number_input(
    "Previous Grade",
    min_value=float(df["previous_grade"].min()),
    max_value=float(df["previous_grade"].max()),
    value=float(df["previous_grade"].median()),
    step=1.0
)

final_exam_score = st.sidebar.number_input(
    "Final Exam Score",
    min_value=float(df["final_exam_score"].min()),
    max_value=float(df["final_exam_score"].max()),
    value=float(df["final_exam_score"].median()),
    step=1.0
)

input_data = pd.DataFrame({
    "gender": [gender],
    "parental_education": [parental_education],
    "internet_access": [internet_access],
    "extracurricular_activities": [extracurricular_activities],
    "part_time_job": [part_time_job],
    "study_time_hours": [study_time_hours],
    "attendance_percent": [attendance_percent],
    "sleep_hours": [sleep_hours],
    "previous_grade": [previous_grade],
    "final_exam_score": [final_exam_score]
})

st.subheader("Student Prediction")

if st.button("Predict Final Grade", type="primary"):

    prediction = best_model.predict(input_data)[0]

    probabilities = best_model.predict_proba(input_data)[0]

    classes = best_model.named_steps[
        "model"
    ].classes_

    probability_df = pd.DataFrame({
        "Grade": classes,
        "Probability": probabilities
    })

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Final Grade",
            prediction
        )

    with col2:
        confidence = probabilities[
            list(classes).index(prediction)
        ] * 100

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

    st.subheader("Prediction Probabilities")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=probability_df,
        x="Grade",
        y="Probability",
        ax=ax
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Final Grade")

    st.pyplot(fig)

st.divider()

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Test Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Macro F1 Score",
        f"{macro_f1:.4f}"
    )

with col3:
    st.metric(
        "Best CV F1 Score",
        f"{grid.best_score_:.4f}"
    )

st.write("Best Random Forest Parameters")

st.json(grid.best_params_)

st.subheader("Cross Validation Results")

cv_df = pd.DataFrame({
    "Model": list(cv_results.keys()),
    "Macro F1 Score": list(cv_results.values())
})

st.dataframe(
    cv_df,
    use_container_width=True
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Final Grade Distribution")

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="final_grade",
        ax=ax
    )

    ax.set_xlabel("Final Grade")
    ax.set_ylabel("Number of Students")

    st.pyplot(fig)

with col2:
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=best_model.named_steps["model"].classes_,
        yticklabels=best_model.named_steps["model"].classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

st.divider()

st.subheader("Classification Report")

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)

st.subheader("Feature Importance")

importance_display = importance_df.copy()

importance_display["Feature"] = (
    importance_display["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)

st.dataframe(
    importance_display.head(15),
    use_container_width=True
)

fig, ax = plt.subplots(figsize=(10, 6))

top_features = importance_display.head(15).sort_values(
    by="Importance"
)

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature",
    ax=ax
)

ax.set_xlabel("Importance")
ax.set_ylabel("Feature")

st.pyplot(fig)

st.divider()

st.subheader("Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

st.write(
    f"Dataset contains **{df.shape[0]} students** and **{df.shape[1]} columns**."
)