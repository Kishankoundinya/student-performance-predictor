
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
