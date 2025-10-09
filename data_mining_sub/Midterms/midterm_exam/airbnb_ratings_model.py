# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime as dt


# %%
current_dir = os.getcwd()
df = pd.read_csv(f'{current_dir}\\Airbnb_Open_Data.csv')
df_copy = df.copy()

df_copy.columns

# %%
columns_to_drop = ['id', 'NAME', 'host id', 'host name', 'neighbourhood group', 'neighbourhood', 'last review', 'lat', 'long', 'country', 'country code', 'Construction year', 'last review', 'house_rules', 'license']
# columns_to_drop = ['id', 'NAME', 'host id', 'host name', 'neighbourhood group', 'neighbourhood', 'last review', 'reviews per month', 'lat', 'long', 'country', 'country code', 'Construction year', 'last review', 'house_rules', 'license']

df_copy.drop(columns=columns_to_drop, inplace=True)
df_copy.isnull().sum()

# %%
df_copy.describe()

# %%
df_copy = df_copy[(df_copy['minimum nights'] >= 0) & (df_copy['minimum nights'] <= 8) & (df_copy['availability 365'] >= 0) & (df_copy['availability 365'] <= 365)]
df_copy['price_per_night'] = df_copy['price'] / df_copy['minimum nights']
df_copy['review_ratio'] = df_copy['number of reviews'] / df_copy['reviews per month'].replace(0, np.nan)
df_copy['is_long_term'] = df_copy['minimum nights'].apply(lambda x: 1 if x > 31 else 0)


# %%
df_copy['number of reviews'] = pd.qcut(df_copy['number of reviews'], q=3, labels=['low','medium','high'])
df_copy.fillna(df_copy.median(numeric_only=True), inplace=True)
df_copy['number of reviews'] = df_copy['number of reviews'].apply(lambda x: 2 if x == 'high' else 1 if x == 'medium' else 0)
for col in df_copy.select_dtypes(include=['object', 'category']).columns:
    df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
    
df_copy.isnull().sum()


# %%
df_copy['instant_bookable'] = df_copy['instant_bookable'].astype(str).str.strip().str.upper()
df_copy['instant_bookable'] = df_copy['instant_bookable'].apply(lambda x: 1 if x == 'TRUE' else 0)

df_copy['instant_bookable'].value_counts()

# %%
df_copy.columns

# %%
dummies_columns = ['host_identity_verified', 'cancellation_policy', 'room type']
df_copy = pd.get_dummies(df_copy, columns=dummies_columns, drop_first=True, dummy_na=False, dtype=int)
df_copy.head(1)

# %%
df_copy.columns

# %%
df_copy.corr()['number of reviews']

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

X = df_copy.drop(columns=['number of reviews'])
y = df_copy['number of reviews']

# Replace inf/-inf with nan, then fill nan with column median
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)

# %%
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
y_pred

# %%
confusion_matrix = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix:\n", confusion_matrix)

accuracy_score = accuracy_score(y_test, y_pred)
print("Confusion Matrix Accuracy Score:", accuracy_score)

f1_score = f1_score(y_test, y_pred, average='weighted')
print("Confusion Matrix F1 Score:", f1_score)

precision_score = precision_score(y_test, y_pred, average='weighted')
print("Precision Score:", precision_score)

recall_score = recall_score(y_test, y_pred, average='weighted')
print("Recall Score:", recall_score)


# %%
import pickle 
filename = 'logistic_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(log_reg, file)

print(f"Model saved as {filename}")

# %%
# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
X_rf = df_copy.drop(columns=['number of reviews', 'reviews per month'])

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y, test_size=0.25, random_state=42, stratify=y)

# #Create a rf Classifier
# rf = RandomForestClassifier()
# rf.fit(X_train_rf, y_train_rf)


# %%
param_dist = {'n_estimators': randint(50,200),
                'max_depth': randint(1, 20)}

# Create a random forest classifier
rf = RandomForestClassifier(random_state=42)

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=5, 
                                    cv=5)

# Fit the random search object to the data
rand_search.fit(X_train_rf, y_train_rf)

best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# %%
# Generate predictions with the best model
y_pred_rf = best_rf.predict(X_test_rf)

# Create the confusion matrix
cm = confusion_matrix(y_test_rf, y_pred_rf)

ConfusionMatrixDisplay(confusion_matrix=cm).plot()

# %%

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", accuracy_score(y_test_rf, y_pred_rf))


# %%
# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train_rf.columns).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar()

# %%
filename = 'random_forest_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(rand_search, file)

print(f"Model saved as {filename}")


