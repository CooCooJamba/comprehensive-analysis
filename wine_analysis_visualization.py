"""
Comprehensive Wine Dataset Analysis and Visualization Script

This script performs exploratory data analysis and visualization on the Wine
dataset using various techniques including statistical analysis, bar charts,
pie charts, line plots, t-SNE, and UMAP dimensionality reduction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.datasets import load_wine
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import fashion_mnist
import umap
import time


def load_and_explore_wine_data():
    """
    Load the Wine dataset and perform initial exploratory data analysis.
    
    Returns:
        tuple: DataFrame containing wine features and target values
    """
    # Load the Wine dataset from sklearn
    wine_data = load_wine()
    
    # Create DataFrame with feature data
    df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
    
    # Create DataFrame with target values
    targets = pd.DataFrame(data=wine_data.target, columns=['target'])
    
    # Display dataset statistics
    print("Dataset Statistics:")
    print(df.describe())
    print("\n")
    
    # Display dataset information
    print("Dataset Information:")
    print(df.info())
    print("\n")
    
    # Display first few rows of the dataset
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\n")
    
    # Check for missing values
    print("Missing values per column:")
    print(df.isnull().sum())
    print("\n")
    
    # Drop any rows with missing values (though Wine dataset typically has none)
    df.dropna(inplace=True)
    
    return df, targets


def create_alcohol_bar_plot(df):
    """
    Create an interactive bar plot showing alcohol content by sample.
    
    Args:
        df (DataFrame): Wine dataset with features
    """
    # Extract alcohol data and sample labels
    data = df['alcohol']  # Quantitative measure
    labels = df.index  # Sample index as labels
    
    # Create bar plot
    fig = go.Figure()
    fig.add_traces(go.Bar(
        x=labels,
        y=data,
        marker=dict(color=data, coloraxis="coloraxis"),
        base=dict(color='black', width=2)
    ))
    
    # Customize plot layout
    fig.update_layout(
        title='Alcohol Content by Sample',
        title_x=0.5,
        title_font_size=20,
        xaxis_title='Sample',
        yaxis_title='Alcohol',
        xaxis_tickangle=315,
        xaxis_tickfont=dict(size=14),
        yaxis_tickfont=dict(size=14),
        height=700,
        showlegend=False
    )
    
    # Add grid styling
    fig.update_yaxes(gridcolor='ivory', gridwidth=2)
    fig.update_xaxes(gridcolor='ivory', gridwidth=2)
    
    # Display the plot
    fig.show()


def create_alcohol_pie_chart(df):
    """
    Create a pie chart showing alcohol content distribution for first 10 samples.
    
    Args:
        df (DataFrame): Wine dataset with features
    """
    # Extract alcohol data
    data = df['alcohol']
    
    # Create pie chart for first 10 samples
    fig = go.Figure(data=[go.Pie(
        labels=df.index, 
        values=data[:10],
        marker=dict(line=dict(color='black', width=2))
    )])
    
    # Customize chart
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(title='Alcohol Content Distribution (First 10 Samples)', title_x=0.5)
    
    # Display the chart
    fig.show()


def create_alcohol_line_plot(df):
    """
    Create a line plot showing alcohol content across samples.
    
    Args:
        df (DataFrame): Wine dataset with features
    """
    # Set figure size
    plt.figure(figsize=(10, 6))
    
    # Create line plot
    plt.plot(df.index, df['alcohol'], marker='o', color='crimson', 
             markerfacecolor='white', markeredgecolor='black', 
             markersize=10, linewidth=2)
    
    # Customize plot
    plt.title('Alcohol Content over Samples')
    plt.grid(color='mistyrose', linestyle='-', linewidth=2)
    plt.xlabel('Sample', fontsize=16)
    plt.ylabel('Alcohol', fontsize=16)
    
    # Display the plot
    plt.show()


def perform_tsne_visualization():
    """
    Perform t-SNE dimensionality reduction on Fashion MNIST dataset
    and visualize results with different perplexity values.
    """
    # Load Fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Preprocess data: reshape and normalize
    x = x_train.reshape((60000, 28 * 28)) / 255.0
    y = y_train
    
    # Define perplexity values to test
    perplexities = [5, 30, 50]
    
    # Create subplots for different perplexity values
    plt.figure(figsize=(20, 5))
    
    for i, perplexity in enumerate(perplexities):
        # Initialize t-SNE with specified perplexity
        tsne = TSNE(perplexity=perplexity, random_state=42)
        
        # Measure execution time
        start_time = time.time()
        
        # Apply t-SNE to first 1000 samples (for faster computation)
        tsne_results = tsne.fit_transform(x[:1000])
        
        # Create subplot
        plt.subplot(1, len(perplexities), i + 1)
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y[:1000], cmap='jet', s=5)
        plt.colorbar()
        plt.title(f't-SNE with Perplexity={perplexity}')
        
        # Print execution time
        print(f"t-SNE execution time (perplexity={perplexity}): {time.time() - start_time:.2f} seconds")
    
    # Adjust layout and display plots
    plt.tight_layout()
    plt.show()


def perform_umap_visualization(df):
    """
    Perform UMAP dimensionality reduction on wine dataset and visualize results.
    
    Args:
        df (DataFrame): Wine dataset with features
    """
    # Measure execution time
    start_time = time.time()
    
    # Initialize UMAP reducer
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    
    # Apply UMAP dimensionality reduction
    umap_result = reducer.fit_transform(df)
    
    # Create scatter plot of UMAP results
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c='green')
    plt.title('UMAP Visualization with n_neighbors=15, min_dist=0.1')
    plt.show()
    
    # Print execution time
    print(f"UMAP execution time: {time.time() - start_time:.2f} seconds")


def main():
    """
    Main function to execute the complete analysis pipeline.
    """
    # Load and explore wine data
    df, targets = load_and_explore_wine_data()
    
    # Create visualizations
    create_alcohol_bar_plot(df)
    create_alcohol_pie_chart(df)
    create_alcohol_line_plot(df)
    
    # Perform dimensionality reduction and visualization
    perform_tsne_visualization()
    perform_umap_visualization(df)


if __name__ == "__main__":
    main()

