from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC # Works same as logistic regression
from sklearn.svm import SVC # Nonlinear classifier default L2 and Hinge loss
SVC(gamma=0.01) # default is kernel="rbf"  Creates a nonlinear model by transforming it to a linear separable plane
SVC(kernel="linear") # Used for linear models
inds_ascending = np.argsort(lr.coef_.flatten())


# Instantiate an RBF SVM
svm = SVC()
parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)


Logistic regression in sklearn:
linear_model.LogisticRegression
C (inverse regularization strength)
penalty (type of regularization)
multi_class (type of multi-class)

SVM in sklearn:
svm.LinearSVC and svm.SVC
Key hyperparameters in sklearn:
C (inverse regularization strength)
kernel (type of kernel)
gamma (inverse RBF smoothness)



SGDClassifier:scaleswelltolargedatasets
from sklearn.linear_model import SGDClassifier
logreg = SGDClassifier(loss='log')
linsvm = SGDClassifier(loss='hinge')
SGDClassifier hyperparameter alpha is like 1/C
