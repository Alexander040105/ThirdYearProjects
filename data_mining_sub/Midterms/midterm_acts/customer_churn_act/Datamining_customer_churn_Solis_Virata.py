# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pydotplus
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os 
from sklearn.metrics import f1_score

# %%
current_dir = os.getcwd()
df = pd.read_csv(f'{current_dir}\\Telco_customer_churn.csv')
df_copy = df.copy()
df.head(5)

# %%
df.columns

# %%
columns_to_drop = ['Lat Long', 'Latitude', 'Longitude', 'Count', 'Streaming TV', 'Streaming Movies', 'Zip Code', 'Churn Reason','CustomerID', 'Country', 'State', 'City']

df_copy.drop(columns=columns_to_drop, inplace=True)
df_copy.head()

# %%
df_copy.isnull().sum()

# %%
df_copy = pd.get_dummies(df_copy, columns=['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Contract', 'Paperless Billing', 'Payment Method'], drop_first=False, dummy_na=False, dtype=int)

df_copy.head(1)

# %%
print(f'Dataframe Columns: {df_copy.columns.tolist()}')

# Replace blanks with NaN
df_copy['Total Charges'] = df_copy['Total Charges'].replace(" ", np.nan)

# Convert column to float
df_copy['Total Charges'] = df_copy['Total Charges'].astype(float)

# Handle missing values (e.g., fill with mean or 0)
df_copy['Total Charges'] = df_copy['Total Charges'].fillna(df_copy['Total Charges'].mean())

# %%
print(f'Dataframe Dtypes: {df_copy.dtypes}')

# %%
leak_cols = ['Churn Value', 'Churn Score', 'CLTV']
feature_cols = [col for col in df_copy.columns if col not in leak_cols + ['Churn Label']]
print(f'Feature Columns: {feature_cols}')

# %%

X = df_copy[feature_cols].dropna()
y = df_copy['Churn Label']
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


# %%
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=5, random_state=1)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print(f'Decision Tree Predictions: {y_pred}')

# %%
# --- Decision Tree Evaluation ---
dt_accuracy = accuracy_score(y_test, y_pred)
print(f'Decision Tree Accuracy: {dt_accuracy:.2f}')

dt_report = classification_report(y_test, y_pred)
print(dt_report)

dt_confusion = confusion_matrix(y_test, y_pred)
print(dt_confusion)

dt_f1_score = f1_score(y_test, y_pred, pos_label='Yes')
print(f'Decision Tree F1 Score: {dt_f1_score:.2f}')


# %%
from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(clf, 
                            out_file=None, 
                            feature_names=X.columns,
                            class_names=y.unique().astype(str),
                            filled=True, rounded=True,
                            special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png")  # Saves as PNG


# %%
#KNN ML ALGORITHM
X = df_copy[feature_cols].dropna()
y = df_copy['Churn Label']
# Split dataset into training set and test set
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X, y, test_size=0.3, random_state=1)
# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_knn)
X_test_scaled = scaler.transform(X_test_knn)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# %%
y_knn_pred = knn.predict(X_test_scaled)
print(f'KNN Predictions: {y_knn_pred}')

# %%
knn_accuracy = accuracy_score(y_test, y_knn_pred)
print(f"Accuracy: {knn_accuracy}")

knn_report = classification_report(y_test_knn, y_knn_pred)
print(knn_report)

knn_f1_score = f1_score(y_test_knn, y_knn_pred, pos_label='Yes')
print(f"F1 Score: {knn_f1_score}") 

knn_confusion_matrix = confusion_matrix(y_test, y_knn_pred)
print("Confusion Matrix:\n", knn_confusion_matrix)


