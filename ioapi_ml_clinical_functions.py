# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:38:33 2023

@author: japostol
"""

import sys
sys.path.append('C:\\Users\\japostol\\Desktop\\SPN Clinical Python codes\\')
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix

import ioapi_ml_clinical_models

#%%

def select_features(data,selected_features):
    new_data = data[selected_features]
    return new_data

def fit(classifier_name,X,y,test,selected_features):
    
    classifiers_needing_label_encoding = ['bayes','knn','logistic','decision_tree','svm','nusvm','lsvm','bagging_meta_estimator','adaboost',
    'catboost','lightgbm','rf','xgb','sgd','mlp']  
    if classifier_name in classifiers_needing_label_encoding:    
        # Label encode the target variable y if it contains non-numerical values
        label_encoder = LabelEncoder()
        yen = label_encoder.fit_transform(y)
        
        # Convert non-numerical columns in X to numerical using one-hot encoding
        Xen = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)
        test = pd.get_dummies(test, columns=test.select_dtypes(include=['object']).columns)
    else:
        Xen = X
        yen = y
        
    Xen = select_features(Xen,selected_features)
    test = select_features(test,selected_features)
    classifier = ioapi_ml_clinical_models.selector(classifier_name)
    classifier.fit(Xen, yen)
    #importance = classifier.feature_importances_
    importance = None
    ypred = classifier.predict(test)
    ypred_proba = classifier.predict_proba(test)
    
    return ypred,yen,classifier,importance, Xen, y,ypred_proba

def grid_search(classifier_name,X,y,test,selected_features):
    
    from sklearn.model_selection import GridSearchCV
    
    classifiers_needing_label_encoding = ['bayes','knn','logistic','decision_tree','svm','nusvm','lsvm','bagging_meta_estimator','adaboost',
    'catboost','lightgbm','rf','xgb','sgd','mlp']    
    if classifier_name in classifiers_needing_label_encoding:    
        # Label encode the target variable y if it contains non-numerical values
        label_encoder = LabelEncoder()
        yen = label_encoder.fit_transform(y)
        
        # Convert non-numerical columns in X to numerical using one-hot encoding
        Xen = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)
        test = pd.get_dummies(test, columns=test.select_dtypes(include=['object']).columns)
    else:
        Xen = X
        yen = y
    Xen = select_features(Xen,selected_features)
    test = select_features(test,selected_features)
    classifier = ioapi_ml_clinical_models.selector(classifier_name)
    
    
    if classifier_name == 'rf':
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 300, 600],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
    
    if classifier_name == 'xgb':
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [200, 400, 600],
            'learning_rate': [0.01, 0.1],
            'max_depth': [ 4, 5, 7, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 1.0]
        }
        
    
    
    if classifier_name == 'catboost':
        # Define hyperparameter grid
        param_grid = {
            'iterations': [50, 100, 200],
            'depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5]
        }
        
    
        
    if classifier_name == 'logistic':
        # Define hyperparameter grid
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga']
        }
            

        
    if classifier_name == 'knn':
        # Define hyperparameter grid
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
        

    if classifier_name == 'lightgbm':
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [ 100, 200, 400, 600],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [5, 10, 15],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }


    if classifier_name == 'svm':
        # Define hyperparameter grid
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto']
        }
        


    if classifier_name == 'adaboost':
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        


    if classifier_name == 'decision_tree':
        # Define hyperparameter grid
        param_grid = {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2', None]
                }
    

    if classifier_name == 'bagging_meta_estimator':
        # Define hyperparameter grid
        param_grid = {
                    'n_estimators': [10, 50, 100],
                    'max_samples': [0.5, 1.0],
                    'max_features': [0.5, 1.0],
                    'bootstrap': [True, False],
                    'bootstrap_features': [True, False]
                    }
        
    if classifier_name == 'nusvm':
        # Define hyperparameter grid
        param_grid = {
            'nu': [0.25, 0.5, 0.75],
            'kernel': ['linear', 'rbf', 'poly'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto']
        }      
        
    if classifier_name == 'svm':
        # Define hyperparameter grid
        param_grid = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }      
        
    if classifier_name == 'lsvm':
        # Define hyperparameter grid
        param_grid = {
            'penalty': ['l1', 'l2'],
            'loss': ['hinge', 'squared_hinge'],
            'C': [0.1, 1, 10]
        }     
        
    if classifier_name == 'mlp':
        # Define hyperparameter grid
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive']
        }
        
        
    if classifier_name == 'sgd':
        # Define hyperparameter grid
        param_grid = {
            'loss': ['hinge', 'log', 'modified_huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'optimal', 'invscaling'],
            'eta0': [0.01, 0.1, 1.0]
        }


        
        
    # Initialize GridSearchCV
    grid_search_model = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='roc_auc')
    
    # Perform grid search
    grid_search_model.fit(Xen, yen)
    
    # Get the best parameters
    best_params = grid_search_model.best_params_
    print("Best Parameters for {}:".format(classifier_name), best_params)
        

    
    return grid_search_model,best_params

def train_kfold (classifier_name,X,y,selected_features):
    
    classifiers_needing_label_encoding = ['bayes','knn','logistic','decision_tree','svm','nusvm','lsvm','bagging_meta_estimator','adaboost',
    'catboost','lightgbm','rf','xgb','sgd','mlp']  
    if classifier_name in classifiers_needing_label_encoding:
        # Label encode the target variable y if it contains non-numerical values
        label_encoder = LabelEncoder()
        yen = label_encoder.fit_transform(y)
        
        # Convert non-numerical columns in X to numerical using one-hot encoding
        Xen = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)
    else:
        Xen = X
        yen = y
    
    Xen = select_features(Xen,selected_features)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_predictions = []
    all_predictions_proba = []
    all_true_labels = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(Xen)):
        X_train, X_test = Xen.iloc[train_idx], Xen.iloc[test_idx]
        y_train, y_test = yen[train_idx], yen[test_idx]
    
        # Fit the model and obtain predictions for the fold
        classifier = ioapi_ml_clinical_models.selector(classifier_name)
        classifier.fit(X_train, y_train)
        fold_predictions = classifier.predict(X_test)
        fold_predictions_proba = classifier.predict_proba(X_test)
        
        # Store predictions and true labels for later evaluation
        all_predictions.extend(fold_predictions)
        all_predictions_proba.extend(fold_predictions_proba)
        all_true_labels.extend(y_test)
    
    
    return all_predictions,all_true_labels,classifier,Xen,yen,all_predictions_proba
    
def metrics(all_predictions,all_true_labels,all_predictions_proba):   
    # Function to calculate accuracy
    def calculate_accuracy(all_predictions, all_true_labels):
        return accuracy_score(all_true_labels, all_predictions)
    
    # Function to calculate sensitivity (recall for class 1)
    def calculate_sensitivity(all_predictions, all_true_labels):
        return recall_score(all_true_labels, all_predictions, pos_label=1)
    
    # Function to calculate specificity (recall for class 0)
    def calculate_specificity(all_predictions, all_true_labels):
        tn, fp, fn, tp = confusion_matrix(all_true_labels, all_predictions).ravel()
        return tn / (tn + fp), tp,tn,fp,fn
    
    # Function to calculate recall (class 1)
    def calculate_recall(all_predictions, all_true_labels):
        return recall_score(all_true_labels, all_predictions)
    
    # Function to calculate precision (positive predictive value)
    def calculate_precision(all_predictions, all_true_labels):
        return precision_score(all_true_labels, all_predictions)
    
    # Function to calculate AUC score
    def calculate_auc_score(all_predictions, all_true_labels):
        return roc_auc_score(all_true_labels, all_predictions)
    
    # Example usage:
    accuracy = calculate_accuracy(all_predictions, all_true_labels)
    sensitivity = calculate_sensitivity(all_predictions, all_true_labels)
    specificity,tp,tn,fp,fn = calculate_specificity(all_predictions, all_true_labels)
    recall = calculate_recall(all_predictions, all_true_labels)
    precision = calculate_precision(all_predictions, all_true_labels)
    auc_score = calculate_auc_score(all_predictions_proba, all_true_labels)
    
    # Display results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    
    
    metrics = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Recall': recall,
        'Precision': precision,
        'AUC': auc_score,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        }
    
    return metrics
    
    
def print_metrics (metrics):
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print ('')
    print ('')
    for key, value in metrics.items():
        print(f"{value}")   
    
    
    
    
    
    
    
    
    
    
    
    
    
    