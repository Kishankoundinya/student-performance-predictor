# Student Performance Predictor

## Team Members

| S. No. | Name                 | Enrollment Number |
| ------ | -------------------- | ----------------- |
| 1      | **FARIYAL FATIMA**   | CSJMA23001390016  |
| 2      | **KISHAN KOUNDINYA** | CSJMA23001390021  |
| 3      | **NAMRATA SINGH**    | CSJMA23001390024  |
| 4      | **NIHARIKA TIWARI**  | CSJMA23001390025  |

---

## Project Overview

Student Performance Predictor is a machine learning-based web application that predicts a student's final grade using academic performance, study habits, attendance, sleep patterns, parental education, internet access, extracurricular activities, and other student-related factors.

The application is developed using Python and provides an interactive web interface through Streamlit.

The project compares Logistic Regression and Random Forest classification models and uses hyperparameter tuning to improve the Random Forest model.

---

## Objectives

The main objectives of this project are:

1. Predict a student's final grade using machine learning.
2. Analyze factors associated with student academic performance.
3. Compare Logistic Regression and Random Forest models.
4. Evaluate models using Macro F1 score and other classification metrics.
5. Perform cross-validation for reliable model evaluation.
6. Optimize the Random Forest model using GridSearchCV.
7. Build an interactive prediction interface using Streamlit.
8. Visualize model performance and feature importance.

---

## Features

### Student Grade Prediction

Users can enter the following student information:

* Gender
* Parental Education
* Internet Access
* Extracurricular Activities
* Part-Time Job
* Study Time Hours
* Attendance Percentage
* Sleep Hours
* Previous Grade
* Final Exam Score

The trained Random Forest model predicts the student's final grade.

### Prediction Confidence

The application displays the confidence of the predicted grade based on the model's predicted probabilities.

### Prediction Probability

A bar chart displays the probability of each possible final grade.

### Model Comparison

The project compares:

* Logistic Regression
* Random Forest

using 5-fold cross-validation with Macro F1 as the evaluation metric.

### Hyperparameter Tuning

Random Forest is optimized using `GridSearchCV`.

The following parameters are evaluated:

* `n_estimators`
* `max_depth`
* `min_samples_leaf`

### Model Evaluation

The application displays:

* Test Accuracy
* Test Macro F1 Score
* Best Cross-Validation F1 Score
* Classification Report
* Confusion Matrix

### Feature Importance

The application displays the most important features used by the optimized Random Forest model.

### Dataset Visualization

The application provides:

* Final Grade Distribution
* Confusion Matrix
* Feature Importance Chart
* Dataset Preview

---

## Machine Learning Workflow

```text
Dataset
   |
   v
Data Loading
   |
   v
Data Cleaning
   |
   v
Feature Selection
   |
   v
Train-Test Split
   |
   v
Data Preprocessing
   |
   v
Model Training
   |
   v
5-Fold Cross Validation
   |
   v
Random Forest Hyperparameter Tuning
   |
   v
Best Model Selection
   |
   v
Test Set Evaluation
   |
   v
Streamlit Prediction Interface
```

---

## Dataset

The application requires the following CSV file:

```text
student_performance_dataset.csv
```

The dataset contains information about students' academic performance and study habits.

### Dataset Columns

| Column                       | Description                                  |
| ---------------------------- | -------------------------------------------- |
| `student_id`                 | Unique student identifier                    |
| `gender`                     | Student gender                               |
| `parental_education`         | Parent's education level                     |
| `internet_access`            | Availability of internet access              |
| `extracurricular_activities` | Participation in extracurricular activities  |
| `part_time_job`              | Whether the student has a part-time job      |
| `study_time_hours`           | Number of hours spent studying               |
| `attendance_percent`         | Student attendance percentage                |
| `sleep_hours`                | Average number of sleeping hours             |
| `previous_grade`             | Previous academic grade                      |
| `final_exam_score`           | Final examination score                      |
| `final_grade`                | Target variable representing the final grade |

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

### Machine Learning Algorithms

#### Logistic Regression

Logistic Regression is used as a baseline classification model for comparison.

#### Random Forest Classifier

Random Forest is used as the primary classification model. It combines multiple decision trees to produce a more robust classification model.

---

## Data Preprocessing

The project uses a `ColumnTransformer` to process numerical and categorical features separately.

### Numerical Features

The following numerical features are standardized using `StandardScaler`:

```text
study_time_hours
attendance_percent
sleep_hours
previous_grade
final_exam_score
```

### Categorical Features

The following categorical features are encoded using `OneHotEncoder`:

```text
gender
parental_education
internet_access
extracurricular_activities
part_time_job
```

The encoder uses:

```python
OneHotEncoder(handle_unknown="ignore")
```

This allows the model to handle unknown categorical values during prediction.

Missing values in `parental_education` are replaced with:

```text
Unknown
```

---

## Train-Test Split

The dataset is divided into:

```text
80% Training Data
20% Testing Data
```

The split uses:

```python
random_state=42
```

and stratification based on the target variable.

---

## Model Evaluation

The project uses **Macro F1 Score** as the primary metric during cross-validation because the target variable contains multiple grade categories.

### Cross-Validation Results

The models were evaluated using 5-fold cross-validation.

| Model               | Macro F1 Score | Standard Deviation |
| ------------------- | -------------: | -----------------: |
| Logistic Regression |          0.759 |            ± 0.061 |
| Random Forest       |      **0.763** |        **± 0.008** |

Random Forest achieved a slightly higher average Macro F1 score than Logistic Regression.

More importantly, Random Forest had a substantially lower standard deviation, indicating more consistent performance across the five validation folds.

---

## Hyperparameter Tuning

The Random Forest model was optimized using `GridSearchCV`.

The following hyperparameters were tested:

```text
n_estimators:
200, 400

max_depth:
None, 8, 12

min_samples_leaf:
1, 2, 4
```

The optimization used:

```text
5-Fold Cross Validation
Scoring: Macro F1
```

The best-performing Random Forest configuration was selected for final testing.

---

## Final Model Results

After hyperparameter tuning, the Random Forest model achieved:

| Metric        |     Result |
| ------------- | ---------: |
| Test Accuracy |    **98%** |
| Test Macro F1 | **0.7828** |

### Classification Report

```text
              precision    recall  f1-score   support

           A       1.00      1.00      1.00        57
           B       0.99      0.99      0.99        71
           C       0.98      0.98      0.98        52
           D       0.90      1.00      0.95        18
           F       0.00      0.00      0.00         2

    accuracy                           0.98       200
   macro avg       0.77      0.79      0.78       200
weighted avg       0.97      0.98      0.98       200
```

### Result Interpretation

The optimized Random Forest model achieved **98% accuracy** on the test dataset.

The model performed particularly well for grades A, B, C, and D.

The F grade received an F1 score of 0.00 because there were only **two F-grade samples** in the test set, and the model did not correctly classify them.

This also explains the difference between the high accuracy and the lower Macro F1 score. Macro F1 gives equal importance to every grade category, whereas accuracy is strongly influenced by the larger classes.

The weighted F1 score of approximately **0.98** indicates excellent overall performance across the test dataset.

---

## Model Evaluation Metrics

### Accuracy

Accuracy measures the percentage of correctly classified samples.

```text
Accuracy = Correct Predictions / Total Predictions
```

The final model achieved:

```text
98% Accuracy
```

### Macro F1 Score

Macro F1 calculates the F1 score for every class independently and then takes the average.

This metric is particularly useful for observing how the model performs across different grade categories.

The final model achieved:

```text
Macro F1 = 0.7828
```

### Confusion Matrix

The confusion matrix shows the number of correct and incorrect predictions for each grade.

### Classification Report

The classification report provides:

* Precision
* Recall
* F1 Score
* Support

for each grade category.

---

## Streamlit Application

The application is built using Streamlit and provides an interactive interface for student grade prediction.

The application contains the following sections:

```text
Student Information
        |
        v
Student Prediction
        |
        v
Prediction Probabilities
        |
        v
Model Performance
        |
        v
Cross Validation Results
        |
        v
Final Grade Distribution
        |
        v
Confusion Matrix
        |
        v
Classification Report
        |
        v
Feature Importance
        |
        v
Dataset Preview
```

---

## Project Structure

```text
student-performance-predictor/
|
├── app.py
├── student_performance_dataset.csv
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
```

Navigate to the project directory:

```bash
cd student-performance-predictor
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

#### macOS / Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

Using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install the libraries manually:

```bash
pip install pandas numpy streamlit matplotlib seaborn scikit-learn
```

---

## Running the Application

Make sure the following files are present in the project directory:

```text
app.py
student_performance_dataset.csv
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

Open the address in a web browser to access the application.

---

## How to Use

### Step 1

Run the Streamlit application.

### Step 2

Enter the student's information using the sidebar.

### Step 3

Click the **Predict Final Grade** button.

### Step 4

The application displays:

* Predicted Final Grade
* Prediction Confidence
* Probability for each grade

### Step 5

Explore the model analysis sections:

* Model Performance
* Cross-Validation Results
* Final Grade Distribution
* Confusion Matrix
* Classification Report
* Feature Importance
* Dataset Preview

---

## Application Outputs

The application generates the following outputs:

### Student Prediction

```text
Predicted Final Grade
Prediction Confidence
```

### Prediction Probability Chart

Displays the probability associated with each final grade.

### Model Performance

Displays:

```text
Test Accuracy
Macro F1 Score
Best CV F1 Score
```

### Cross-Validation Results

Displays the Macro F1 score of:

* Logistic Regression
* Random Forest

### Final Grade Distribution

Displays the distribution of grades within the dataset.

### Confusion Matrix

Displays actual versus predicted grade classifications.

### Classification Report

Displays precision, recall, F1 score, and support for each grade.

### Feature Importance

Displays the features that contribute most to the Random Forest predictions.

### Dataset Preview

Displays the first 10 rows of the dataset and the total number of students and columns.

---

## Key Machine Learning Concepts

This project demonstrates the practical implementation of:

* Supervised Machine Learning
* Multi-Class Classification
* Logistic Regression
* Random Forest
* Train-Test Split
* Feature Preprocessing
* Standard Scaling
* One-Hot Encoding
* Machine Learning Pipelines
* Cross Validation
* Hyperparameter Tuning
* GridSearchCV
* Model Evaluation
* Classification Report
* Confusion Matrix
* Feature Importance
* Data Visualization
* Streamlit Application Development

---

## Future Enhancements

Possible improvements to the project include:

* Adding additional classification algorithms such as SVM, XGBoost, and Gradient Boosting.
* Adding personalized academic recommendations.
* Adding historical performance tracking.
* Adding downloadable prediction reports.
* Deploying the application online.
* Adding user authentication.
* Connecting the application to a database.
* Implementing explainable AI using SHAP or LIME.
* Improving the Streamlit user interface.
* Adding interactive data filtering and visualization.

---

## Limitations

* Model performance depends on the quality and representativeness of the dataset.
* The F grade has very few samples, which affects its classification performance and the overall Macro F1 score.
* Predictions are based only on the features available in the dataset.
* Dataset imbalance can influence model performance.
* The model is retrained when the Streamlit application starts.
* The predictions should be considered machine learning estimates rather than definitive academic evaluations.
