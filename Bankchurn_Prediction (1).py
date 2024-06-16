import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("/content/bank cusomer churn prediction.csv")

print(df.columns)
print(df.head())
print(df.tail())
print(df.info)
print(df.shape)

cols = ['Gender', 'Geography', 'HasCrCard', 'IsActiveMember', 'Exited']
existing_cols = [col for col in cols if col in df.columns]

plt.figure(figsize=(20, 4))
for i, col in enumerate(existing_cols):
    ax = plt.subplot(1, len(existing_cols), i+1)
    sns.countplot(x=col, data=df)
    ax.set_title(f"{col}")
plt.show()
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Unnamed: 14'])

# Check for missing values
print(df.isnull().sum())

# Handle missing values for features
imputer = SimpleImputer(strategy='mean')
df['CreditScore'] = imputer.fit_transform(df[['CreditScore']])
df['Age'] = imputer.fit_transform(df[['Age']])
df['Tenure'] = imputer.fit_transform(df[['Tenure']])
df['Balance'] = imputer.fit_transform(df[['Balance']])
df['NumOfProducts'] = imputer.fit_transform(df[['NumOfProducts']])
df['EstimatedSalary'] = imputer.fit_transform(df[['EstimatedSalary']])

# Handle missing values for the target variable
df['Exited'] = df['Exited'].fillna(df['Exited'].mode()[0])

# Encode categorical features
cat_features = ['Gender', 'Geography', 'HasCrCard', 'IsActiveMember']
for col in cat_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Scale numerical features
scaler = StandardScaler()
num_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
df[num_features] = scaler.fit_transform(df[num_features])

# Split the data into training and testing sets
X = df.drop(['Exited'], axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(df.isnull().sum())

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))

print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))
# Confusion Matrix for Random Forest as an example
conf_matrix = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()

print("Logistic Regression Report:\n", classification_report(y_test, y_pred_lr))
# Confusion Matrix for Logistic Regression as an example
conf_matrix = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

print("Gradient Boosting Report:\n", classification_report(y_test, y_pred_gb))
# Confusion Matrix for Gradient Boosting as an example
conf_matrix = confusion_matrix(y_test, y_pred_gb)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Gradient Boosting')
plt.show()
