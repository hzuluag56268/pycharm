import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://raw.githubusercontent.com/datacamp/machine-learning-with-scikit-learn-live-training/master/data/telco_churn.csv'
telco = pd.read_csv(url)
pd.set_option('display.max_columns', None)
#print(telco.head())
#print(telco.info())
#print(telco.nunique())
telco.drop(['customerID', 'Unnamed: 0'], axis=1, inplace= True)
telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors= 'coerce')
# telco['TotalCharges'] = telco['TotalCharges'].astype('float64', errors='ignore')
# sns.distplot(telco['TotalCharges'])
# plt.show()
telco.loc[telco['TotalCharges'].isna(), 'TotalCharges'] = telco['TotalCharges'].median()
telco['InternetService'].replace('dsl', 'DSL', inplace= True)

features   = [column_name for column_name in telco.columns if column_name != 'Churn']
categorical= [column_name for column_name in features if telco[column_name].dtype == 'object']
numeric    = [column_name for column_name in features if column_name not in categorical]
'''fig, axes = plt.subplots(2, 2)
for ax, column_name in zip(axes.flatten(), categorical[0:4]):
    sns.countplot(column_name, hue='Churn', data=telco, ax=ax)
fig, axes = plt.subplots(2, 2)
for ax, column_name in zip(axes.flatten(), categorical[4:8]):
    sns.countplot(column_name, hue='Churn', data=telco, ax=ax)
fig, axes = plt.subplots(2, 3)
for ax, column_name in zip(axes.flatten(), categorical[8:]):
    sns.countplot(column_name, hue='Churn', data=telco, ax=ax)
plt.show()''' #plot categorical
'''fig, axes = plt.subplots(1,3)
for ax, column_name in zip(axes.flatten(), numeric):
    sns.boxplot(x='Churn', y=column_name, data=telco, ax=ax )
plt.show()'''#plot numerical
# preprocessing
X = telco[features]
y = telco['Churn'].replace({'Stayed': 0, 'Churned': 1})

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=.25, random_state=123)

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(train_X[numeric])
train_numeric_transform = scalar.transform(train_X[numeric])
test_numeric_transform = scalar.transform(test_X[numeric])
train_X[numeric] = train_numeric_transform
test_X[numeric] = test_numeric_transform
train_X = pd.get_dummies(train_X, columns=categorical, drop_first=True)
test_X = pd.get_dummies(test_X, columns=categorical, drop_first=True)
# creating new feature
test_X['Churn'] = test_y
train_X['Churn'] = train_y
service_columns = ['OnlineSecurity_Yes', 'OnlineBackup_Yes',
                   'DeviceProtection_Yes', 'TechSupport_Yes']
train_X['in_ecosystem'] = train_X[service_columns].sum(axis=1)
train_X['in_ecosystem'] = np.where(train_X['in_ecosystem'] > 1, 1, 0)
test_X['in_ecosystem'] = test_X[service_columns].sum(axis=1)
test_X['in_ecosystem'] = np.where(test_X['in_ecosystem'] > 1, 1, 0)
train_X.drop('Churn', axis=1, inplace=True)
test_X.drop('Churn', axis=1, inplace=True)
#modeling
'''from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(train_X, train_y)
pred_test_Y = knn.predict(test_X)
pred_train_Y = knn.predict(train_X)
test_accuracy = accuracy_score(test_y, pred_test_Y)
train_accuracy = accuracy_score(train_y, pred_train_Y)
print('test accuracy: {:2f} \ntrain_accuracy: {:2f}'.format(test_accuracy, train_accuracy))
''' #using knn
'''from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
dec_tree = DecisionTreeClassifier(random_state=123)
rand_forest = RandomForestClassifier(random_state=123)
dec_tree.fit(train_X, train_y)
rand_forest.fit(train_X, train_y)
test_pred_y_tree = dec_tree.predict(test_X)
train_pred_y_tree = dec_tree.predict(train_X)
test_pred_y_forest = rand_forest.predict(test_X)
train_pred_y_forest = rand_forest.predict(train_X)

test_accuracy_tree = accuracy_score(test_y, test_pred_y_tree)
train_accuracy_tree = accuracy_score(train_y, train_pred_y_tree)
test_accuracy_forest = accuracy_score(test_y, test_pred_y_forest)
train_accuracy_forest = accuracy_score(train_y, train_pred_y_forest)
print('accuracy_test_tree {:8f} \naccuracy_train_tree: {:1f} \naccuracy_test_forest: {:1f}'
      '\naccuracy_train_forest: {:1f}'.format(test_accuracy_tree, train_accuracy_tree,
                                       test_accuracy_forest, train_accuracy_forest)) ''' # tree and randomforest
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


dec_tree = DecisionTreeClassifier(random_state=123, max_depth=4, max_features=25)
cv_scores = cross_val_score(dec_tree, train_X, train_y, cv=10)
dec_tree.fit(train_X, train_y)

test_pred_y_tree = dec_tree.predict(test_X)
train_pred_y_tree = dec_tree.predict(train_X)

test_accuracy_tree = accuracy_score(test_y, test_pred_y_tree)
train_accuracy_tree = accuracy_score(train_y, train_pred_y_tree)
print('cv_scores: {} \nmean_cv_scores: {:2f} \ntest_accuracy_tree: {:2f} \ntrain_accuracy_tree: {:2f}'
      .format(cv_scores, np.mean(cv_scores), test_accuracy_tree, train_accuracy_tree))

#df = pd.DataFrame({'labels': train_X.columns, "value": dec_tree.feature_importances_})
#print(df.sort_values('value', ascending=False))
''' # tree and crossval
#tuning gridsearch

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


dec_tree = DecisionTreeClassifier(random_state=123)
params = {'max_depth': [6, 8, 10, 12],
          'max_features': [20, 25, 30]}
clf = GridSearchCV(dec_tree, params, cv=10, verbose=2)
clf.fit(train_X, train_y)
y_pred = clf.predict(test_X)

print('best params: ', clf.best_params_)
print('accuracy', round(accuracy_score(test_y, y_pred), 4))