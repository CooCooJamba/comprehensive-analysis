#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Drug Classification Analysis
This script performs classification analysis on the Drug200 dataset using
multiple machine learning models including Logistic Regression, KNN, and SVM.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, metrics, linear_model, neighbors, svm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_and_explore_data(file_path):
    """
    Load the dataset and perform initial exploration.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Display basic information about the data
    print('Data Head:')
    print(data.head(), '\n')
    print('Data Description:')
    print(data.describe(), '\n')
    
    # Display unique values in target variable
    print('Unique Drugs:', data['Drug'].unique())
    
    # Display value counts for categorical features
    print('\nBlood Pressure Value Counts:')
    print(data['BP'].value_counts())
    print('\nDrug Value Counts:')
    print(data['Drug'].value_counts())
    
    return data


def preprocess_data(data, target_name='Drug'):
    """
    Preprocess the data by splitting into train/test sets and scaling features.
    
    Args:
        data (pandas.DataFrame): Input data
        target_name (str): Name of the target column
        
    Returns:
        tuple: Processed training and testing data (X_train, X_test, y_train, y_test)
    """
    # Identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = data.select_dtypes(exclude=["object"]).columns.tolist()
    
    # Remove target from categorical columns
    categorical_cols.remove(target_name)
    
    print(f'\nCategorical Columns: {categorical_cols}')
    print(f'Numerical Columns: {numeric_cols}')
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        data.drop(target_name, axis=1),
        data[target_name],
        test_size=0.3,
        random_state=1,
        shuffle=True
    )
    
    print(f'\nTraining shapes - X: {X_train.shape}, y: {y_train.shape}')
    print(f'Testing shapes - X: {X_test.shape}, y: {y_test.shape}')
    
    # Scale numerical features
    scaler = preprocessing.StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train[numeric_cols])
    X_test_num_scaled = scaler.transform(X_test[numeric_cols])
    
    # Encode categorical features
    encoder = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_cat = encoder.fit_transform(X_train[categorical_cols]).astype(int)
    X_test_cat = encoder.transform(X_test[categorical_cols]).astype(int)
    
    # Combine numerical and categorical features
    X_train_processed = np.hstack((X_train_num_scaled, X_train_cat))
    X_test_processed = np.hstack((X_test_num_scaled, X_test_cat))
    
    return X_train_processed, X_test_processed, y_train, y_test


def evaluate_model(y_test, y_pred, class_names):
    """
    Evaluate classification model performance with multiple metrics.
    
    Args:
        y_test (array): True labels
        y_pred (array): Predicted labels
        class_names (list): List of class names
    """
    # Classification report
    print('Classification Report:')
    print(metrics.classification_report(y_test, y_pred))
    
    # Confusion matrix
    print('Confusion Matrix:')
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Binarize labels for multi-class ROC and Precision-Recall curves
    y_test_bin = preprocessing.label_binarize(y_test, classes=class_names)
    y_pred_bin = preprocessing.label_binarize(y_pred, classes=class_names)
    n_classes = len(class_names)
    
    # Precision-Recall curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for i in range(n_classes):
        precision, recall, _ = metrics.precision_recall_curve(y_test_bin[:, i], y_pred_bin[:, i])
        pr_auc = metrics.auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'Class {i} (AUC = {pr_auc:0.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    # ROC curve
    plt.subplot(1, 2, 2)
    for i in range(n_classes):
        fpr, tpr, _ = metrics.roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:0.2f})')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()


def train_logistic_regression(X_train, y_train, X_test, y_test, class_names):
    """
    Train and evaluate a Logistic Regression model.
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        X_test (array): Testing features
        y_test (array): Testing labels
        class_names (list): List of class names
    """
    print('\n=== Logistic Regression ===')
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    evaluate_model(y_test, y_pred, class_names)


def train_knn(X_train, y_train, X_test, y_test, class_names):
    """
    Train and evaluate a K-Nearest Neighbors model with hyperparameter tuning.
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        X_test (array): Testing features
        y_test (array): Testing labels
        class_names (list): List of class names
    """
    print('\n=== K-Nearest Neighbors ===')
    knn = neighbors.KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 26)}
    
    grid_search = model_selection.GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    print(f'Best parameters: {grid_search.best_params_}')
    y_pred = grid_search.predict(X_test)
    
    evaluate_model(y_test, y_pred, class_names)


def train_svm(X_train, y_train, X_test, y_test, class_names):
    """
    Train and evaluate a Support Vector Machine model with hyperparameter tuning.
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        X_test (array): Testing features
        y_test (array): Testing labels
        class_names (list): List of class names
    """
    print('\n=== Support Vector Machine ===')
    svm_model = svm.SVC()
    param_grid = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
    
    grid_search = model_selection.GridSearchCV(svm_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    print(f'Best parameters: {grid_search.best_params_}')
    y_pred = grid_search.predict(X_test)
    
    print(f'Accuracy: {grid_search.score(X_test, y_test):.4f}')
    evaluate_model(y_test, y_pred, class_names)


def main():
    """Main function to execute the drug classification analysis."""
    # Load and explore the data
    data = load_and_explore_data('drug200.csv')
    
    # Visualize target distribution
    plt.figure(figsize=(10, 5))
    plt.bar(data['Drug'].unique(), data['Drug'].value_counts())
    plt.title('Drug Distribution')
    plt.xlabel('Drug Type')
    plt.ylabel('Count')
    plt.show()
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    class_names = ['DrugY', 'drugC', 'drugX', 'drugA', 'drugB']
    
    # Train and evaluate models
    train_logistic_regression(X_train, y_train, X_test, y_test, class_names)
    train_knn(X_train, y_train, X_test, y_test, class_names)
    train_svm(X_train, y_train, X_test, y_test, class_names)


if __name__ == "__main__":
    main()

