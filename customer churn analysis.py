import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df= pd.read_csv("C:/Users/maddy/Downloads/Churn_Modelling.csv")

df.head()
df.info()
df.isnull().sum()
df[df.duplicated()]
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df = pd.get_dummies(df,columns=['Geography'], drop_first=True)
df.head()
features = ['CreditScore','Gender','Age','Tenure','Balance','NumOfProducts',
            'HasCrCard','IsActiveMember','EstimatedSalary','Geography_Germany','Geography_Spain']
x= df[features]
y= df['Exited']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2, random_state=42)
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)
x _train[:5], x_test[:5]
# finding acuracy using RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(conf_matrix)
print(class_report)
print(accuracy)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
names = [features[i] for i in indices]

plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.barh(range(x.shape[1]), importances[indices])
plt.yticks(range(x.shape[1]), names)
plt.show()

#finding Accuracy using Logistic Regression
from sklearn.linear_model import LogisticRegression

# Build and train the Logistic Regression Model
log_reg  = LogisticRegression(random_state=42)
log_reg.fit(x_train, y_train)

# Make predictions
y_pred_log_reg = log_reg.predict(x_test)

# Evaluate the model
confu_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
class_report_log_reg = classification_report(y_test, y_pred_log_reg)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

print(confu_matrix_log_reg, class_report_log_reg, accuracy_log_reg )

# Finding accuracy using SVM
from sklearn.svm import SVC

# Build and train the SVM Model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(x_train, y_train)

# Make predictions
y_pred_svm = svm_model.predict(x_test)

# Evaluate the model
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
class_report_svm = classification_report(y_test, y_pred_svm)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(conf_matrix_svm, class_report_svm, accuracy_svm)

#Finding accuracy using KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

# Build and train the model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)

#Make predictions 
y_pred_knn = knn_model.predict(x_test)

# Evaluate the model
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
class_reoport_knn = classification_report(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(conf_matrix_knn, class_reoport_knn, accuracy_knn)           

# Finding accuracy using Gradient Boosting Classifier 
from sklearn.ensemble import GradientBoostingClassifier

# Build and train the model
gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_model.fit(x_train, y_train)

# Make predictions
y_pred_gbm = gbm_model.predict(x_test)

# Evaluate the model 
conf_matrix_gbm = confusion_matrix(y_test, y_pred_gbm)
class_report_gbm = classification_report(y_test, y_pred_gbm)
accuracy_gbm = accuracy_score(y_test, y_pred_gbm)
print(conf_matrix_gbm, class_report_gbm, accuracy_gbm)

df = pd.read_csv("C:/Users/maddy/Downloads/Churn_Modelling.csv")

# Binary feature for balance
df['BalanceZero'] = (df['Balance'] == 0).astype(int)

# Age groups
df['AgeGroups'] = pd.cut(df['Age'], bins =[18,25,35,45,55,65,75,85,95], labels = ['18-25', '26-35', '36-45', '46-55','56-65','66-75','76-85','86-95'])

# Balance to salary ratio
df['BalanceToSalaryRatio'] = df['Balance']/df['EstimatedSalary']

# Interaction features between NumOfProducts and IsActiveMember
df['ProductUsage'] = df['NumOfProducts'] * df['IsActiveMember']

# Tenure grouping
df['TenureGroup'] = pd.cut(df['Tenure'], bins =[0,2,5,7,10], labels =['0-2','3-5','6-7','8-10'])
scalar = StandardScaler()

label_encoder = LabelEncoder()
df['Gender']= label_encoder.fit_transform(df['Gender'])
df= pd.get_dummies(df,columns=['Geography'], drop_first=True)
df['Male_Germany']= df['Gender']*df['Geography_Germany']
df['Male_Spain']=df['Gender']*df['Geography_Spain']

df = pd.get_dummies(df, columns =['AgeGroups','TenureGroup'], drop_first= True)
features = ['CreditScore','Age','Tenure','Balance','NumOfProducts',
            'HasCrCard','IsActiveMember','EstimatedSalary','Geography_Germany','Geography_Spain','BalanceZero','BalanceToSalaryRatio',
           'ProductUsage','Male_Germany','Male_Spain'] + [col for col in df.columns if 'Agegroups_' in col or 'TenureGroup_' in col]
x= df[features]
y= df['Exited']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(conf_matrix)
print(class_report)
print(accuracy)

