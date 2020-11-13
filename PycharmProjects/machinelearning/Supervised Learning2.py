'''Classification'''

from sklearn import datasets
plt.style.use('ggplot')
iris = datasets.load_iris()
iris.keys()

pd.plotting.scatter_matrix(df, c=y, figsize=[8, 8], s=150, marker='D')

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
#These were feature names ,Also used for grid search

from sklearn.svm import SVC
svc = SVC

df.drop('party', axis=1).values


'''Regression'''

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg,X,y,cv=5) #Returns score values of the
# model which for default is R2 in linear regression

from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1, normalize=True)
ridge.alpha  #method to modify its parameters
#Penalizes large coefficients
#L2

from sklearn.linear_model import Lasso #Perform Ridge first before lasso
lasso = Lasso(alpha=0.1, normalize=True)
#Very good at feature selection Via  .coef_
lasso_coef = lasso.fit(X,y).coef_
#The order of the columnsname of X matches the order of .coef_
# L1


from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(l1_ratio= from 0 to 1)
#uses a combination of L1 and L2
'''Fine-tuning your model'''

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

logreg = LogisticRegression().fit()
y_pred_prob = logreg.predict_proba(X_test)[:,1] #Uses probabilities

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_metric = roc_auc_score(y_test, y_pred_prob)


from sklearn.model_selection import GridSearchCV

pd.get_dummies(df,drop_first=True)
df[df == '?'] = np.nan

'''Preprocessing and pipelines'''

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler #used this scalar for pipeline