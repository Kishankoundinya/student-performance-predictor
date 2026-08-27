
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
