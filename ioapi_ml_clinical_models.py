# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:52:01 2023

@author: japostol
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 09:38:03 2021

@author: japostol
"""

from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier


def selector (classifier_name):
    classifier = None
    if classifier_name == 'catboost':
        classifier = catboost()
    if classifier_name == 'logistic':
        classifier = logistic_regression_classifier()
    if classifier_name == 'bayes':
        classifier = naive_bayes_classifier()
    if classifier_name == 'knn':
        classifier = k_nn_classifier()
    if classifier_name == 'rf':
        classifier = random_forest_classifier()
    if classifier_name == 'xgb':
        classifier = xgboost_classifier_scikit()
    if classifier_name == 'lightgbm':
        classifier = lightgbm_classifier()
    if classifier_name == 'svm':
        classifier = svm_classifier()
    if classifier_name == 'adaboost':
        classifier = adaboost_classifier()
    if classifier_name == 'decision_tree':
        classifier = decision_tree_classifier()
    if classifier_name == 'bagging_meta_estimator':
        classifier = bagging_meta_estimator()
    if classifier_name == 'nusvm':
        classifier = nu_svm_classifier()
    if classifier_name == 'lsvm':
        classifier = linear_svm_classifier()
    if classifier_name == 'sgd':
        classifier = sgd_classifier()        
    if classifier_name == 'mlp':
        classifier = neural_network_classifier()          

    return classifier


def catboost():
    classifier = CatBoostClassifier(depth= 5, iterations= 200, l2_leaf_reg= 3, learning_rate= 0.01)
    return classifier


def xgboost_classifier_scikit ():
    from sklearn.ensemble import GradientBoostingClassifier
    
    # for GRID-SEARCH based on AUC: learning_rate= 0.1, max_depth= 7, min_samples_leaf= 2, min_samples_split= 5, n_estimators= 600, subsample= 0.8
    # for GRID-SEARCH based on Sensitivity: learning_rate= 0.1, max_depth= 10, min_samples_leaf= 4, min_samples_split= 5, n_estimators= 200, subsample= 1
    # for GRID-SEARCH based on Accuracy: learning_rate= 0.1, max_depth= 5, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 100, subsample= 0.8
    
    
    classifier = GradientBoostingClassifier(learning_rate = 0.1, 
                                            max_depth = 4, 
                                            min_samples_leaf = 2, 
                                            min_samples_split = 4, 
                                            n_estimators = 400, 
                                            subsample = 0.8)
    
    
    
    return classifier


def logistic_regression_classifier():
    return LogisticRegression()


def naive_bayes_classifier():
    return GaussianNB(var_smoothing=1e-09)


def k_nn_classifier():
    return KNeighborsClassifier(n_neighbors= 7, p= 1, weights= 'distance')


def random_forest_classifier():
    """
    This function returns an instance of the RandomForestClassifier from scikit-learn with specified parameters.

    Returns:
        RandomForestClassifier: A Random Forest classifier with specified parameters.
    """
    # Creating an instance of the RandomForestClassifier with specified parameters
    classifier = RandomForestClassifier(
        n_estimators=100,           # The number of trees in the forest
        criterion='gini',           # Function to measure the quality of a split ('gini' or 'entropy')
        max_depth=None,             # The maximum depth of the trees
        min_samples_split=2,        # The minimum number of samples required to split an internal node
        min_samples_leaf=1,         # The minimum number of samples required to be at a leaf node
        max_features='auto',        # The number of features to consider when looking for the best split
        bootstrap=True,             # Whether bootstrap samples are used when building trees
        random_state=None,          # Seed for the random number generator
        n_jobs=None,                # The number of jobs to run in parallel for both fit and predict
        verbose=0,                  # Controls the verbosity when fitting and predicting
        class_weight=None,          # Weights associated with classes in the form {class_label: weight}
        warm_start=False,           # When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble
        ccp_alpha=0.0               # Complexity parameter used for Minimal Cost-Complexity Pruning
    )
    
    return classifier



def lightgbm_classifier():
    
    # GRID SEARC for Recall: Best Parameters for lightgbm: {'colsample_bytree': 1.0, 'learning_rate': 0.2, 'max_depth': 10, 'n_estimators': 200, 'reg_alpha': 0, 'reg_lambda': 0.1, 'subsample': 0.8}
    
    return LGBMClassifier(colsample_bytree= 1, learning_rate= 0.2, max_depth= 10, n_estimators= 200, reg_alpha= 0, reg_lambda= 0.1, subsample= 0.8)


def svm_classifier():
    # Grid search for recall: Best Parameters for svm: {'C': 0.1, 'degree': 2, 'gamma': 'scale', 'kernel': 'linear'}
    
    return SVC(probability=True,C= 0.1, degree= 2, gamma= 'scale', kernel= 'linear')


def adaboost_classifier():
    """
    This function returns an instance of the AdaBoostClassifier from scikit-learn with specified parameters.

    Returns:
        AdaBoostClassifier: An AdaBoost classifier with specified parameters.
    """
    # Creating an instance of the base estimator
    base_estimator = DecisionTreeClassifier(max_depth=1)

    # Creating an instance of the AdaBoostClassifier with specified parameters
    classifier = AdaBoostClassifier(
        base_estimator=base_estimator,   # The base estimator from which the boosted ensemble is built
        n_estimators=200,                # The maximum number of estimators at which boosting is terminated
        learning_rate=1.0,              # Weight applied to each classifier at each boosting iteration
        algorithm='SAMME.R',            # Algorithm used for boosting ('SAMME' or 'SAMME.R')
        random_state=None               # Seed for the random number generator
    )
    
    return classifier





###################

def decision_tree_classifier():
    """
    This function returns an instance of the DecisionTreeClassifier from scikit-learn with specified parameters.

    Returns:
        DecisionTreeClassifier: A decision tree classifier with specified parameters.
    """
    # Creating an instance of the DecisionTreeClassifier with specified parameters
    classifier = DecisionTreeClassifier(
        criterion='gini',         # Criterion for splitting (could be 'gini' or 'entropy')
        splitter='best',          # Strategy for choosing the split at each node ('best' or 'random')
        max_depth=None,           # Maximum depth of the tree
        min_samples_split=2,      # Minimum number of samples required to split an internal node
        min_samples_leaf=1,       # Minimum number of samples required to be at a leaf node
        max_features=None,        # Number of features to consider when looking for the best split
        random_state=None         # Seed for the random number generator
    )
    
    return classifier

def bagging_meta_estimator():
    """
    This function returns an instance of the BaggingClassifier from scikit-learn with specified parameters.

    Returns:
        BaggingClassifier: A bagging meta-estimator with specified parameters.
    """
    # Creating an instance of the base estimator
    base_estimator = DecisionTreeClassifier()

    # Creating an instance of the BaggingClassifier with specified parameters
    classifier = BaggingClassifier(
        base_estimator=base_estimator,  # The base estimator to fit on random subsets of the dataset
        n_estimators=10,                # The number of base estimators in the ensemble
        max_samples=1.0,                # The number of samples to draw from X to train each base estimator
        max_features=1.0,               # The number of features to draw from X to train each base estimator
        bootstrap=True,                 # Whether samples are drawn with replacement
        bootstrap_features=False,       # Whether features are drawn with replacement
        n_jobs=None,                    # The number of jobs to run in parallel
        random_state=None               # Seed for the random number generator
    )
    
    return classifier


def nu_svm_classifier():
    """
    This function returns an instance of the NuSVC classifier from scikit-learn with specified parameters.

    Returns:
        NuSVC: A Nu-Support Vector Classification classifier with specified parameters.
    """
    # Creating an instance of the NuSVC classifier with specified parameters
    classifier = NuSVC(
        nu=0.5,                    # An upper bound on the fraction of margin errors
        kernel='rbf',              # Specifies the kernel type to be used in the algorithm
        degree=3,                  # Degree of the polynomial kernel function ('poly')
        gamma='scale',             # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        coef0=0.0,                 # Independent term in kernel function
        shrinking=True,            # Whether to use the shrinking heuristic
        probability=False,         # Whether to enable probability estimates
        tol=1e-3,                  # Tolerance for stopping criterion
        cache_size=200,            # Size of the kernel cache (in MB)
        class_weight=None,         # Set the parameter C of class i to class_weight[i]*C for SVC
        verbose=False,             # Enable verbose output
        max_iter=-1,               # Hard limit on iterations within solver, or -1 for no limit
        decision_function_shape='ovr',  # One-vs-one ('ovo') or one-vs-rest ('ovr') decision function
        random_state=None          # Seed for the random number generator
    )
    
    return classifier


def linear_svm_classifier():
    """
    This function returns an instance of the LinearSVC classifier from scikit-learn with specified parameters.

    Returns:
        LinearSVC: A Linear Support Vector Classification classifier with specified parameters.
    """
    # Creating an instance of the LinearSVC classifier with specified parameters
    classifier = LinearSVC(
        penalty='l2',              # Specifies the norm used in the penalization ('l1' or 'l2')
        loss='squared_hinge',      # Specifies the loss function ('hinge' or 'squared_hinge')
        dual=True,                 # Dual or primal formulation (True for n_samples > n_features)
        tol=1e-4,                  # Tolerance for stopping criteria
        C=1.0,                     # Regularization parameter
        multi_class='ovr',         # Determines the multi-class strategy ('ovr' or 'crammer_singer')
        fit_intercept=True,        # Whether to calculate the intercept for this model
        intercept_scaling=1,       # When self.fit_intercept is True, instance vector x becomes [x, self.intercept_scaling]
        class_weight=None,         # Set the parameter C of class i to class_weight[i]*C for SVC
        verbose=0,                 # Enable verbose output
        random_state=None,         # Seed for the random number generator
        max_iter=1000              # The maximum number of iterations to be run
    )
    
    return classifier


def sgd_classifier():
    """
    This function returns an instance of the SGDClassifier from scikit-learn with specified parameters.

    Returns:
        SGDClassifier: A Stochastic Gradient Descent classifier with specified parameters.
    """
    # Creating an instance of the SGDClassifier with specified parameters
    classifier = SGDClassifier(
        loss='hinge',              # Loss function to be used ('hinge' is for linear SVM)
        penalty='l2',              # The penalty (regularization term) to be used ('l2', 'l1', or 'elasticnet')
        alpha=0.0001,              # Constant that multiplies the regularization term
        l1_ratio=0.15,             # The Elastic Net mixing parameter (only used if penalty is 'elasticnet')
        fit_intercept=True,        # Whether the intercept should be estimated or not
        max_iter=1000,             # The maximum number of passes over the training data (epochs)
        tol=1e-3,                  # The stopping criterion
        shuffle=True,              # Whether or not the training data should be shuffled after each epoch
        verbose=0,                 # The verbosity level
        epsilon=0.1,               # Epsilon in the epsilon-insensitive loss functions
        n_jobs=None,               # The number of CPUs to use to do the OVA (One Versus All, for multi-class problems) computation
        random_state=None,         # Seed for the random number generator
        learning_rate='optimal',   # The learning rate schedule
        eta0=0.0,                  # The initial learning rate for the 'constant', 'invscaling' or 'adaptive' schedules
        power_t=0.5,               # The exponent for inverse scaling learning rate
        early_stopping=False,      # Whether to use early stopping to terminate training when validation score is not improving
        validation_fraction=0.1,   # The proportion of training data to set aside as validation set for early stopping
        n_iter_no_change=5,        # Number of iterations with no improvement to wait before early stopping
        class_weight=None,         # Preset class weights
        warm_start=False,          # Whether to reuse the solution of the previous call to fit as initialization
        average=False              # When set to True, computes the averaged SGD weights
    )
    
    return classifier


def neural_network_classifier():
    """
    This function returns an instance of the MLPClassifier from scikit-learn with specified parameters.

    Returns:
        MLPClassifier: A Multilayer Perceptron classifier with specified parameters.
    """
    # Creating an instance of the MLPClassifier with specified parameters
    classifier = MLPClassifier(
        hidden_layer_sizes=(100,),    # The ith element represents the number of neurons in the ith hidden layer
        activation='relu',            # Activation function for the hidden layer ('identity', 'logistic', 'tanh', 'relu')
        solver='adam',                # The solver for weight optimization ('lbfgs', 'sgd', 'adam')
        alpha=0.0001,                 # L2 penalty (regularization term) parameter
        batch_size='auto',            # Size of minibatches for stochastic optimizers
        learning_rate='constant',     # Learning rate schedule for weight updates ('constant', 'invscaling', 'adaptive')
        learning_rate_init=0.001,     # The initial learning rate used
        power_t=0.5,                  # The exponent for inverse scaling learning rate
        max_iter=200,                 # Maximum number of iterations
        shuffle=True,                 # Whether to shuffle samples in each iteration
        random_state=None,            # Seed for the random number generator
        tol=1e-4,                     # Tolerance for the optimization
        verbose=False,                # Whether to print progress messages to stdout
        warm_start=False,             # When set to True, reuse the solution of the previous call to fit as initialization
        momentum=0.9,                 # Momentum for gradient descent update
        nesterovs_momentum=True,      # Whether to use Nesterov's momentum
        early_stopping=False,         # Whether to use early stopping to terminate training when validation score is not improving
        validation_fraction=0.1,      # The proportion of training data to set aside as validation set for early stopping
        beta_1=0.9,                   # Exponential decay rate for estimates of first moment vector in adam
        beta_2=0.999,                 # Exponential decay rate for estimates of second moment vector in adam
        epsilon=1e-8,                 # Value for numerical stability in adam
        n_iter_no_change=10,          # Maximum number of epochs to not meet tol improvement
        max_fun=15000                 # Maximum number of function calls
    )
    
    return classifier





















