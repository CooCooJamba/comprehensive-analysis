#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Linear Regression Analysis with Gradient Descent and ANOVA Testing

This script performs linear regression analysis using both scikit-learn and manual gradient descent,
and conducts ANOVA and post-hoc tests on a dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def calculate_correlation():
    """
    Calculate and visualize correlation between street and garage data points.
    """
    # Sample data for correlation analysis
    street = [80, 98, 75, 91, 78]
    garage = [100, 82, 105, 89, 102]
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(street, garage)[0, 1]
    print(f"Correlation coefficient: {correlation:.4f}")
    
    # Create scatter plot
    plt.scatter(street, garage)
    plt.xlabel('Street')
    plt.ylabel('Garage')
    plt.title('Scatter Plot: Street vs Garage')
    plt.grid(True)
    plt.show()


def linear_regression_analysis():
    """
    Perform linear regression analysis on gym membership data.
    """
    # Load dataset
    data = pd.read_csv('https://raw.githubusercontent.com/zahid607/Tidyverse/refs/heads/main/gym_members_exercise_tracking.csv')
    
    # Data exploration
    print("Dataset head:")
    print(data.head())
    print("\nDataset info:")
    print(data.info())
    print("\nMissing values:")
    print(data.isna().sum())
    print("\nDescriptive statistics:")
    print(data.describe())
    
    # Select numerical columns
    df = data.select_dtypes(include=['int', 'float'])
    print("\nNumerical data:")
    print(df.head())
    
    # Correlation analysis
    corr = df.corr()['Session_Duration (hours)'].to_frame().round(2)
    print("\nCorrelation with Session Duration:")
    print(corr.style.background_gradient(cmap='coolwarm'))
    
    # Linear regression with scikit-learn
    X = df[['Session_Duration (hours)']]
    y = df['Calories_Burned']
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    model = LinearRegression()
    model.fit(X, y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    print(f"\nScikit-learn model: y = {slope:.6f}x + {intercept:.6f}")
    
    # Plot scikit-learn model
    model_y_sk = slope * X + intercept
    
    plt.figure(figsize=(10, 6))
    plt.plot(X, model_y_sk, linewidth=2, color="r", 
             label=f'Scikit-learn model: y = {slope:.2f}x + {intercept:.2f}')
    plt.scatter(X, y, alpha=0.7)
    plt.grid()
    plt.xlabel('Session Duration (hours)')
    plt.ylabel('Calories Burned')
    plt.legend(prop={'size': 12})
    plt.title('Linear Regression: Scikit-learn Implementation')
    plt.show()
    
    return X, y, slope, intercept


def mse_error(X, w1, w0, y):
    """
    Calculate Mean Squared Error for linear regression.
    
    Parameters:
    X (array): Feature matrix
    w1 (float): Slope parameter
    w0 (float): Intercept parameter
    y (array): Target values
    
    Returns:
    float: Mean squared error
    """
    y_pred = w1 * X[:, 0] + w0
    return np.sum((y - y_pred) ** 2) / len(y_pred)


def gradient_mse(X, w1, w0, y):
    """
    Calculate gradient of Mean Squared Error for linear regression.
    
    Parameters:
    X (array): Feature matrix
    w1 (float): Slope parameter
    w0 (float): Intercept parameter
    y (array): Target values
    
    Returns:
    array: Gradient vector [intercept_gradient, slope_gradient]
    """
    y_pred = w1 * X[:, 0] + w0
    intercept_grad = 2/len(X) * np.sum((y - y_pred)) * (-1)
    slope_grad = 2/len(X) * np.sum((y - y_pred) * (-X[:, 0]))
    return np.array([intercept_grad, slope_grad])


def gradient_descent(X, y, learning_rate=0.01, eps=0.0001, max_iter=100000):
    """
    Perform gradient descent to optimize linear regression parameters.
    
    Parameters:
    X (array): Feature matrix
    y (array): Target values
    learning_rate (float): Step size for gradient descent
    eps (float): Convergence threshold
    max_iter (int): Maximum number of iterations
    
    Returns:
    tuple: Optimized parameters (slope, intercept)
    """
    # Initialize parameters
    w1 = 0
    w0 = 0
    
    # Gradient descent iteration
    for i in range(max_iter):
        current_w1 = w1
        current_w0 = w0
        
        # Calculate gradient and update parameters
        grad = gradient_mse(X, current_w1, current_w0, y)
        w0 = current_w0 - learning_rate * grad[0]
        w1 = current_w1 - learning_rate * grad[1]
        
        # Print progress
        if i % 10000 == 0:
            print(f"Iteration: {i}")
            print(f"Current parameters: slope={current_w1:.6f}, intercept={current_w0:.6f}")
            print(f"MSE: {mse_error(X, current_w1, current_w0, y):.6f}")
            print("----------------------------------------------------------")
        
        # Check for convergence
        if (abs(current_w1 - w1) <= eps) and (abs(current_w0 - w0) <= eps):
            print(f"Converged after {i} iterations")
            break
    
    return w1, w0


def anova_analysis():
    """
    Perform ANOVA analysis on insurance data.
    """
    # Load insurance data
    data = pd.read_csv('insurance.csv')
    print("Insurance data head:")
    print(data.head())
    
    # Get unique regions
    unique_regions = data['region'].unique()
    print("\nUnique regions:", unique_regions)
    
    # Prepare data for ANOVA
    df = pd.DataFrame({"region": data.region, "bmi": data.bmi})
    groups = df.groupby("region").groups
    
    # Extract BMI values for each region
    southwest = data["bmi"][groups["southwest"]]
    southeast = data["bmi"][groups["southeast"]]
    northwest = data["bmi"][groups["northwest"]]
    northeast = data["bmi"][groups["northeast"]]
    
    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(southwest, southeast, northwest, northeast)
    print(f"\nOne-way ANOVA results: F-statistic={f_stat:.4f}, p-value={p_value:.4f}")
    
    # ANOVA using statsmodels
    model = ols('bmi ~ region', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("\nANOVA table:")
    print(anova_table)
    
    # Post-hoc tests (Tukey HSD)
    tukey = pairwise_tukeyhsd(data['bmi'], data['region'])
    print("\nTukey HSD results:")
    print(tukey)
    
    # Visualize Tukey HSD results
    tukey.plot_simultaneous()
    plt.title("Tukey HSD Post-hoc Test")
    plt.show()
    
    # Two-way ANOVA (region and sex)
    model = ols('bmi ~ region + sex', data=data).fit()
    two_way_anova = sm.stats.anova_lm(model, typ=2)
    print("\nTwo-way ANOVA (region + sex):")
    print(two_way_anova)
    
    # Post-hoc test for two-way ANOVA
    data['combination'] = data.region + "/ " + data.sex
    tukey_2way = pairwise_tukeyhsd(endog=data['bmi'], groups=data['combination'], alpha=0.05)
    print("\nTukey HSD for two-way ANOVA:")
    print(tukey_2way)
    
    # Visualize two-way ANOVA results
    tukey_2way.plot_simultaneous()
    plt.title("Tukey HSD Post-hoc Test for Two-way ANOVA")
    plt.show()


if __name__ == "__main__":
    # Part 1: Correlation analysis
    print("=" * 50)
    print("PART 1: CORRELATION ANALYSIS")
    print("=" * 50)
    calculate_correlation()
    
    # Part 2: Linear regression analysis
    print("\n" + "=" * 50)
    print("PART 2: LINEAR REGRESSION ANALYSIS")
    print("=" * 50)
    X, y, sk_slope, sk_intercept = linear_regression_analysis()
    
    # Manual gradient descent
    print("\n" + "=" * 50)
    print("GRADIENT DESCENT OPTIMIZATION")
    print("=" * 50)
    gd_slope, gd_intercept = gradient_descent(X, y)
    print(f"Gradient descent results: slope={gd_slope:.6f}, intercept={gd_intercept:.6f}")
    
    # Compare models
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print(f"Scikit-learn model: y = {sk_slope:.6f}x + {sk_intercept:.6f}")
    print(f"Gradient descent model: y = {gd_slope:.6f}x + {gd_intercept:.6f}")
    
    # Plot both models
    plt.figure(figsize=(10, 6))
    
    # Scikit-learn model
    model_y_sk = sk_slope * X + sk_intercept
    plt.plot(X, model_y_sk, linewidth=2, color="r", 
             label=f'Scikit-learn: y = {sk_slope:.2f}x + {sk_intercept:.2f}')
    
    # Gradient descent model
    x_range = np.arange(0, 3, 0.1)
    gd_model_y = gd_slope * x_range + gd_intercept
    plt.plot(x_range, gd_model_y, '--g', linewidth=2, 
             label=f'Gradient descent: y = {gd_slope:.2f}x + {gd_intercept:.2f}')
    
    plt.scatter(X, y, alpha=0.7)
    plt.grid()
    plt.xlabel('Session Duration (hours)')
    plt.ylabel('Calories Burned')
    plt.legend(prop={'size': 12})
    plt.title('Comparison of Linear Regression Models')
    plt.show()
    
    # Part 3: ANOVA analysis
    print("\n" + "=" * 50)
    print("PART 3: ANOVA ANALYSIS")
    print("=" * 50)
    anova_analysis()

