# Bank Customer Churn analysis

This project predicts whether a bank customer will churn (`Exited` column) using supervised learning. It includes baseline models and engineered features to try to improve predictive performance.



## Files

- `churn_analysis.py` (or the Jupyter notebook version) — main script containing the code you provided.
- `Churn_Modelling.csv` — dataset (not included in repo).
- `README.md` — this file.

---

## Dataset

You'll need the `Churn_Modelling.csv` file. The script expects it at:

C:/Users/maddy/Downloads/Churn_Modelling.csv


(Adjust the path if your dataset is elsewhere.)

Typical important columns in the dataset: `CustomerId`, `Surname`, `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Exited`.

---

## Environment & Requirements

Create a virtual environment and install the required packages. Example using `pip`:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn
Tested with:

Python 3.8+

pandas

numpy

matplotlib

seaborn

scikit-learn


Loading data

Reads CSV into a pandas DataFrame.

Basic preprocessing

Checks for nulls and duplicates.

Encodes Gender with LabelEncoder.

One-hot encodes Geography with pd.get_dummies(..., drop_first=True).

Feature engineering

Adds these derived features (later in the script):

BalanceZero: binary indicator whether Balance == 0.

AgeGroups: categorical age bins (e.g. '18-25', '26-35', ...).

BalanceToSalaryRatio: Balance / EstimatedSalary.

ProductUsage: NumOfProducts * IsActiveMember.

TenureGroup: tenure bins (e.g. '0-2','3-5',...).

Interaction terms: Male_Germany, Male_Spain.

One-hot encodes AgeGroups and TenureGroup.

Note: The script constructs two modeling runs — a simpler features set first, then a second run that includes engineered features.

Train/Test split & scaling

Uses train_test_split(test_size=0.2, random_state=42).

Scales features using StandardScaler.

Models trained

Random Forest (RandomForestClassifier)

Logistic Regression (LogisticRegression)

Support Vector Machine with linear kernel (SVC(kernel='linear'))

K-Nearest Neighbors (KNeighborsClassifier)

Gradient Boosting (GradientBoostingClassifier)

For each model: fit on training data, predict on test data, print confusion_matrix, classification_report, and accuracy_score.

Evaluation & Visualization

Prints model metrics.

Plots a horizontal bar chart of feature importances from Random Forest.

How to run

Put Churn_Modelling.csv at the path referenced or update the script with the correct path.

Save your Python script as churn_analysis.py (or use a Jupyter notebook).

Run:

python churn_analysis.py


Or run cell-by-cell in a notebook.
