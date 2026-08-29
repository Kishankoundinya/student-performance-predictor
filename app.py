
import os
import pandas as pd

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import kagglehub

from google.colab import drive
drive.mount('/content/drive')



import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

RANDOM_STATE = 42

df = pd.read_csv("/content/sample_data/student_performance_dataset.csv")

print(df.shape)
df.head()

df.info()

df.describe(include="all").T

#TASK2

df.isnull().sum().sort_values(ascending=False)

df["parental_education"] = df["parental_education"].fillna("Unknown")


print("Duplicate rows:", df.duplicated().sum())
print("Duplicate student_id:", df["student_id"].duplicated().sum())
for col in ["study_time_hours", "attendance_percent", "sleep_hours", "previous_grade", "final_exam_score"]:
    print(f"{col}: min={df[col].min()}, max={df[col].max()}")



order = ["A", "B", "C", "D", "F"]

ax = sns.countplot(
    data=df,
    x="final_grade",
    hue="final_grade",
    legend=False,
    order=order,
    palette="viridis"
)

ax.set_title("Final Grade Distribution")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom")
plt.show()

numeric_cols = ["study_time_hours", "attendance_percent", "sleep_hours", "previous_grade", "final_exam_score"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.flat, numeric_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
    ax.set_title(col)
axes.flat[-1].axis("off")
plt.tight_layout()
plt.show()

corr = df[numeric_cols].corr()
plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.flat, numeric_cols):
    sns.boxplot(
        data=df,
        x="final_grade",
        y=col,
        hue="final_grade",
        legend=False,
        order=order,
        ax=ax,
        palette="viridis"
    )
    ax.set_title(f"{col} by final_grade")

axes.flat[-1].axis("off")
plt.tight_layout()
plt.show()

cat_cols = ["gender", "parental_education", "internet_access", "extracurricular_activities", "part_time_job"]
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for ax, col in zip(axes.flat, cat_cols):
    ct = pd.crosstab(df[col], df["final_grade"], normalize="index")[order]
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis", legend=False)
    ax.set_title(col)
    ax.set_ylabel("Proportion")
axes.flat[-1].axis("off")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, title="final_grade", loc="lower right")
plt.tight_layout()
plt.show()

#Task 3(Feature Engineering)
feature_cols = [c for c in df.columns if c not in ["student_id", "final_grade"]]
X = df[feature_cols].copy()
y = df["final_grade"].copy()

numeric_features = ["study_time_hours", "attendance_percent", "sleep_hours",
                     "previous_grade", "final_exam_score"]
categorical_features = ["gender", "parental_education", "internet_access",
                         "extracurricular_activities", "part_time_job"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(X_train.shape, X_test.shape)


#TASK 4
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300),
}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1_macro")
    print(f"{name}: macro-F1 = {scores.mean():.3f} (+/- {scores.std():.3f})")

rf_pipe = Pipeline([("prep", preprocessor), ("model", RandomForestClassifier(random_state=RANDOM_STATE))])

param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 8, 12],
    "model__min_samples_leaf": [1, 2, 4],
}

grid = GridSearchCV(rf_pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV macro-F1:", grid.best_score_)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average="macro", zero_division=0))
print()
print(classification_report(y_test, y_pred, labels=order, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=order)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=order, yticklabels=order)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

ohe_cols = best_model.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(ohe_cols)

importances = best_model.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances}) \
    .sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(8, 6))
sns.barplot(
    data=imp_df,
    y="feature",
    x="importance",
    hue="feature",
    legend=False,
    palette="viridis"
)
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()


