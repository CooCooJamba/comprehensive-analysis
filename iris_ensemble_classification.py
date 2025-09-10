"""
Iris Flower Classification using Ensemble Methods
This script demonstrates the application of bagging (Random Forest) and 
boosting (CatBoost) ensemble methods on the Iris dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import catboost as cb


def load_and_preprocess_data():
    """
    Load and preprocess the Iris dataset.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Load Iris dataset
    X, y = load_iris(return_X_y=True)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Print dataset shapes for verification
    print("Full dataset shape:", X.shape)
    print("Training features shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Testing features shape:", X_test.shape)
    print("Testing labels shape:", y_test.shape)
    print()
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier using GridSearchCV for hyperparameter tuning.
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        
    Returns:
        RandomForestClassifier: Best trained model
    """
    # Initialize Random Forest classifier
    random_forest = RandomForestClassifier(random_state=42)
    
    # Define hyperparameter grid for tuning
    params_grid = {
        "max_depth": [12, 18],
        "min_samples_leaf": [3, 10],
        "min_samples_split": [6, 12],
    }
    
    # Initialize GridSearchCV for hyperparameter optimization
    grid_search_random_forest = GridSearchCV(
        estimator=random_forest,
        param_grid=params_grid,
        scoring="f1_macro",
        cv=4,
        n_jobs=-1
    )
    
    # Perform grid search with timing
    print("Training Random Forest with GridSearchCV...")
    start = time.time()
    grid_search_random_forest.fit(X_train, y_train)
    training_time = time.time() - start
    print(f"Training completed in: {training_time:.2f} seconds")
    
    # Get the best model from grid search
    best_model = grid_search_random_forest.best_estimator_
    
    return best_model


def train_catboost(X_train, y_train):
    """
    Train a CatBoost classifier.
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        
    Returns:
        CatBoostClassifier: Trained model
    """
    # Initialize CatBoost classifier
    cb_clf = cb.CatBoostClassifier(verbose=0, random_state=42)
    
    # Train the model with timing
    print("Training CatBoost classifier...")
    start = time.time()
    cb_clf.fit(X_train, y_train)
    training_time = time.time() - start
    print(f"Training completed in: {training_time:.2f} seconds")
    
    return cb_clf


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Evaluate model performance on training and test sets using F1 score.
    
    Args:
        model: Trained classifier
        X_train (array): Training features
        y_train (array): Training labels
        X_test (array): Testing features
        y_test (array): Testing labels
        model_name (str): Name of the model for display purposes
    """
    print(f"\n{model_name} Evaluation:")
    
    # Make predictions on training data
    y_pred_train = model.predict(X_train)
    train_f1 = f1_score(y_train, y_pred_train, average='macro')
    print(f'Training F1 score: {train_f1:.4f}')
    
    # Make predictions on test data
    y_pred_test = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro')
    print(f'Test F1 score: {test_f1:.4f}')
    
    return test_f1


def main():
    """Main function to execute the iris classification pipeline."""
    print("Iris Flower Classification using Ensemble Methods")
    print("=================================================")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Train and evaluate Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_score = evaluate_model(
        rf_model, X_train, y_train, X_test, y_test, "Random Forest"
    )
    
    # Train and evaluate CatBoost
    cb_model = train_catboost(X_train, y_train)
    cb_score = evaluate_model(
        cb_model, X_train, y_train, X_test, y_test, "CatBoost"
    )
    
    # Compare model performance
    print("\nModel Comparison:")
    print(f"Random Forest Test F1: {rf_score:.4f}")
    print(f"CatBoost Test F1: {cb_score:.4f}")
    
    if rf_score > cb_score:
        print("Random Forest performed better.")
    elif cb_score > rf_score:
        print("CatBoost performed better.")
    else:
        print("Both models performed equally.")


if __name__ == "__main__":
    main()

