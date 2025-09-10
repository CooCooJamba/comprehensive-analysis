"""
Comprehensive Data Analysis Script for Insurance and COVID-19 Datasets

This script performs statistical analysis, visualization, and hypothesis testing
on insurance data and COVID-19 case data. It includes descriptive statistics,
distribution analysis, confidence intervals, normality tests, and various
statistical tests including t-tests and chi-square tests.

Requirements:
- numpy
- pandas
- matplotlib
- seaborn
- scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


def load_and_explore_insurance_data():
    """
    Load and perform initial exploration of insurance dataset
    
    Returns:
        pandas.DataFrame: Loaded insurance data
    """
    # Load insurance data
    data = pd.read_csv("insurance.csv")
    
    # Display basic information about the dataset
    print("Insurance Data Overview:")
    print(data.head())
    print("\nData Description:")
    print(data.describe())
    
    return data


def visualize_distributions(data):
    """
    Create histograms for numerical columns in the dataset
    
    Args:
        data (pandas.DataFrame): Insurance dataset
    """
    # Define numerical columns for analysis
    num_cols = ['age', 'bmi', 'children', 'charges']
    
    # Create histograms for numerical columns
    data[num_cols].hist(edgecolor='black', bins=15, figsize=(10, 6))
    plt.suptitle('Distribution of Numerical Variables in Insurance Data')
    plt.tight_layout()
    plt.show()


def calculate_central_tendency(data):
    """
    Calculate and display measures of central tendency and dispersion
    
    Args:
        data (pandas.DataFrame): Insurance dataset
    """
    # Calculate mean and standard deviation for BMI and charges
    bmi_mean = data['bmi'].mean()
    bmi_std = data['bmi'].std()
    charges_mean = data['charges'].mean()
    charges_std = data['charges'].std()

    # Print results
    print(f'BMI - Mean: {bmi_mean:.2f}, Standard Deviation: {bmi_std:.2f}')
    print(f'Charges - Mean: {charges_mean:.2f}, Standard Deviation: {charges_std:.2f}')


def plot_distributions_with_stats(data):
    """
    Create histograms with mean and standard deviation indicators
    
    Args:
        data (pandas.DataFrame): Insurance dataset
    """
    # Calculate statistics
    bmi_mean = data['bmi'].mean()
    bmi_std = data['bmi'].std()
    charges_mean = data['charges'].mean()
    charges_std = data['charges'].std()
    
    # Create subplots
    plt.figure(figsize=(12, 6))

    # BMI distribution
    plt.subplot(1, 2, 1)
    plt.hist(data['bmi'], bins=15, alpha=0.7, color='blue')
    plt.axvline(bmi_mean, color='red', label='Mean')
    plt.axvline(bmi_mean + bmi_std, color='green', linestyle='dashed', label='Standard Deviation')
    plt.axvline(bmi_mean - bmi_std, color='green', linestyle='dashed')
    plt.title('BMI Distribution')
    plt.legend()

    # Charges distribution
    plt.subplot(1, 2, 2)
    plt.hist(data['charges'], bins=15, alpha=0.7, color='orange')
    plt.axvline(charges_mean, color='red', label='Mean')
    plt.axvline(charges_mean + charges_std, color='green', linestyle='dashed', label='Standard Deviation')
    plt.axvline(charges_mean - charges_std, color='green', linestyle='dashed')
    plt.title('Charges Distribution')
    plt.legend()

    plt.tight_layout()
    plt.show()


def create_boxplots(data):
    """
    Create boxplots for numerical variables
    
    Args:
        data (pandas.DataFrame): Insurance dataset
    """
    # Define numerical columns
    num_cols = ['age', 'bmi', 'children', 'charges']
    
    # Create boxplots
    fig, axs = plt.subplots(len(num_cols), 1, figsize=(8, 20))

    for i, column in enumerate(num_cols):
        axs[i].boxplot(data[column], vert=False)
        axs[i].grid()
        axs[i].set_title(f'Box-Plot for {column}')
        axs[i].set_ylabel(column)

    plt.tight_layout()
    plt.show()


def demonstrate_central_limit_theorem(data):
    """
    Demonstrate Central Limit Theorem using sampling
    
    Args:
        data (pandas.DataFrame): Insurance dataset
    """
    # Set parameters for sampling
    sample_sizes = [5, 10, 30, 50, 100]  # Various sample sizes
    num_samples = 300  # Number of samples to generate

    # For each sample size
    for n in sample_sizes:
        sample_means = []  # List to store sample means

        # Generate samples
        for _ in range(num_samples):
            # Random sample from the population
            sample = np.random.choice(data["charges"], size=n)
            sample_means.append(np.mean(sample))  # Add mean to list

        # Convert to numpy array
        sample_means = np.array(sample_means)

        # Calculate mean and standard deviation
        mean = np.mean(sample_means)
        std = np.std(sample_means)

        # Print statistics
        print(f'Sample size: {n}, Mean of sample means: {mean:.2f}, Std of sample means: {std:.2f}')

        # Plot histogram
        plt.figure(figsize=(8, 5))
        sns.histplot(sample_means, kde=True, stat="density", bins=30)
        plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.2f}')
        plt.axvline(mean + std, color='green', linestyle='dashed', linewidth=1, label=f'Std: {std:.2f}')
        plt.axvline(mean - std, color='green', linestyle='dashed', linewidth=1)
        plt.title(f'Distribution of Sample Means (n={n})')
        plt.xlabel('Sample Means')
        plt.ylabel('Density')
        plt.legend()
        plt.show()


def calculate_confidence_intervals(data):
    """
    Calculate confidence intervals for charges and BMI
    
    Args:
        data (pandas.DataFrame): Insurance dataset
    
    Returns:
        tuple: Confidence intervals for charges and BMI at 95% and 99% confidence levels
    """
    def confidence_interval(data, confidence=0.95):
        """
        Calculate confidence interval for a dataset
        
        Args:
            data (array-like): Data to calculate CI for
            confidence (float): Confidence level (default: 0.95)
        
        Returns:
            tuple: Lower and upper bounds of confidence interval
        """
        n = len(data)
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)  # Sample standard deviation
        z = stats.norm.ppf((1 + confidence) / 2)  # Z-score for given confidence level
        margin_error = z * (std_dev / np.sqrt(n))  # Margin of error
        return (mean - margin_error, mean + margin_error)

    # Calculate confidence intervals
    ci_expenses_95 = confidence_interval(data['charges'], 0.95)
    ci_expenses_99 = confidence_interval(data['charges'], 0.99)
    ci_bmi_95 = confidence_interval(data['bmi'], 0.95)
    ci_bmi_99 = confidence_interval(data['bmi'], 0.99)

    # Print results
    print("95% Confidence Interval for charges:", ci_expenses_95)
    print("99% Confidence Interval for charges:", ci_expenses_99)
    print("95% Confidence Interval for BMI:", ci_bmi_95)
    print("99% Confidence Interval for BMI:", ci_bmi_99)
    
    return ci_expenses_95, ci_expenses_99, ci_bmi_95, ci_bmi_99


def test_normality(data):
    """
    Perform normality tests on charges and BMI data
    
    Args:
        data (pandas.DataFrame): Insurance dataset
    """
    # Test for normality for both charges and BMI
    for variable, name in zip([data['charges'], data['bmi']], ['charges', 'BMI']):
        # Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.kstest(variable, 'norm', args=(np.mean(variable), np.std(variable, ddof=1)))
        print(f"KS-test for {name}: statistic = {ks_statistic:.4f}, p-value = {p_value:.4f}")

        # Q-Q plot
        stats.probplot(variable, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot for {name}')
        plt.show()


def load_and_clean_covid_data():
    """
    Load and clean COVID-19 case data
    
    Returns:
        pandas.DataFrame: Cleaned COVID-19 data
    """
    # Load COVID-19 data
    data = pd.read_csv("ECDCCases.csv")
    
    # Display basic information
    print("COVID-19 Data Overview:")
    print(data.head())
    
    # Check missing values
    print("\nMissing Values Percentage:")
    print((data.isnull().sum()/data.count().sum()*100).round(2))
    
    # Drop unnecessary columns
    data = data.drop('Cumulative_number_for_14_days_of_COVID-19_cases_per_100000', axis=1)
    data = data.drop("geoId", axis=1)
    
    # Fill missing values
    data["countryterritoryCode"].fillna("other", inplace=True)
    data['popData2019'].fillna(data['popData2019'].median(), inplace=True)
    
    # Check missing values after cleaning
    print("\nMissing Values Percentage After Cleaning:")
    print((data.isnull().sum()/data.count().sum()*100).round(2))
    
    # Display dataset description
    print("\nCOVID-19 Data Description:")
    print(data.describe())
    
    # Check for records with high death counts
    print("\nRecords with Deaths > 3000:")
    print(data[['countriesAndTerritories', 'deaths', 'day', 'month', 'year']][data['deaths'] > 3000])
    
    print(f"\nCount of records with deaths > 3000: {data[data['deaths'] > 3000].count()['deaths']}")
    
    # Check for duplicates
    print(f"\nDuplicate rows found: {data.duplicated().sum()}")
    
    # Remove duplicates
    data.drop_duplicates(inplace=True)
    print(f"Duplicate rows after cleaning: {data.duplicated().sum()}")
    
    return data


def perform_t_test():
    """
    Perform t-test on BMI data from different regions
    """
    # Load BMI data
    bmi = pd.read_csv("bmi.csv")
    
    # Extract BMI data for northwest and southwest regions
    nbmi = bmi['bmi'][bmi['region'] == 'northwest']
    sbmi = bmi['bmi'][bmi['region'] == 'southwest']
    
    # Perform statistical tests
    t_test = stats.ttest_ind(nbmi, sbmi)
    shapiro_northwest = stats.shapiro(nbmi)
    shapiro_southwest = stats.shapiro(sbmi)
    bartlett_test = stats.bartlett(nbmi, sbmi)
    
    # Print results
    print("T-test Results:", t_test)
    print("Shapiro-Wilk Test (northwest):", shapiro_northwest)
    print("Shapiro-Wilk Test (southwest):", shapiro_southwest)
    print("Bartlett Test:", bartlett_test)


def perform_chi_square_dice_test():
    """
    Perform chi-square test for dice roll data
    """
    # Observed dice roll frequencies
    observed = [97, 98, 109, 95, 97, 104]
    expected = [100] * 6  # Expected uniform distribution
    
    # Chi-square test
    chi2_statistic, p_value = stats.chisquare(observed, f_exp=expected)
    
    print("Chi-Square Test Results for Dice Rolls:")
    print(f"Statistic: {chi2_statistic:.4f}, p-value: {p_value:.4f}")


def perform_chi_square_relationship_test():
    """
    Perform chi-square test for relationship status and employment data
    """
    # Create dataframe with relationship status and employment data
    data = pd.DataFrame({
        'Married': [89, 17, 11, 43, 22, 1],
        'Cohabiting': [80, 22, 20, 35, 6, 4],
        'Single': [35, 44, 35, 6, 8, 22]
    })

    data.index = ['Full-time', 'Part-time', 'Temporarily unemployed', 
                 'Homemaker', 'Retired', 'Student']

    # Chi-square test of independence
    chi2_statistic, p_value, dof, expected = stats.chi2_contingency(data)

    print("Chi-Square Test Results for Relationship Status and Employment:")
    print(f"Statistic: {chi2_statistic:.4f}, p-value: {p_value:.4f}, degrees of freedom: {dof}")


def main():
    """Main function to execute all analyses"""
    print("=" * 60)
    print("INSURANCE DATA ANALYSIS")
    print("=" * 60)
    
    # Insurance data analysis
    insurance_data = load_and_explore_insurance_data()
    visualize_distributions(insurance_data)
    calculate_central_tendency(insurance_data)
    plot_distributions_with_stats(insurance_data)
    create_boxplots(insurance_data)
    demonstrate_central_limit_theorem(insurance_data)
    calculate_confidence_intervals(insurance_data)
    test_normality(insurance_data)
    
    print("\n" + "=" * 60)
    print("COVID-19 DATA ANALYSIS")
    print("=" * 60)
    
    # COVID-19 data analysis
    covid_data = load_and_clean_covid_data()
    
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)
    
    # Statistical tests
    perform_t_test()
    print()
    perform_chi_square_dice_test()
    print()
    perform_chi_square_relationship_test()


if __name__ == "__main__":
    main()

