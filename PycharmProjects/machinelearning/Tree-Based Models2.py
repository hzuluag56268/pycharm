'''1
Classification and Regression Trees'''

#A classification tree divides the feature space into rectangular regions
from sklearn.tree import DecisionTreeClassifier

'''2The Bias-Variance Tradeoff'''


# Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier

# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators=classifiers)

'''3 Bagging and Random Forests'''

# Import BaggingClassifier
from sklearn.ensemble import BaggingClassifier
dt = DecisionTreeClassifier(random_state=1)
bc = BaggingClassifier(base_estimator=dt, n_estimators=50,
                       oob_score=True, random_state=1)
bc.oob_score_ #

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=25,
            random_state=2)
rf.feature_importances_

'''4 Boosting'''
#given that this dataset is imbalanced, you'll be using the ROC AUC score as a metric instead of accuracy.



from sklearn.ensemble import AdaBoostClassifier
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)

.
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=200, max_depth=4,random_state=2)

sgbr = GradientBoostingRegressor(max_depth=4,subsample=.9,max_features=.75,
                                 n_estimators=200,random_state=2)
#To make a stochastic we need to set  suvb sample which is percentage of samples made without
# replacement and Max features  which is sample features made without replacement


'''5 Model Tuning'''

best_model = grid_dt.best_estimator_

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',
                       cv=3,
                       verbose=1,
                       n_jobs=-1)