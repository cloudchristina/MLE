"""
Exploratory Data Analysis (EDA) Demo
=====================================
A beginner-friendly demonstration of EDA techniques:
- Univariate Analysis (single variable)
- Bivariate Analysis (two variables)
- Multivariate Analysis (multiple variables)
- Data visualization and pattern discovery

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================================
# GENERATE SAMPLE E-COMMERCE DATASET
# ============================================================================
# Creating a realistic dataset for EDA practice
# ============================================================================

def generate_ecommerce_data(n_records=500):
    """
    Generate sample e-commerce dataset for EDA practice
    Includes: sales, customers, products, and geographic data
    """
    np.random.seed(42)  # For reproducibility

    # Product categories and price ranges
    products = {
        'Laptop': (800, 2000),
        'Phone': (400, 1200),
        'Tablet': (300, 900),
        'Headphones': (50, 300),
        'Watch': (100, 500),
        'Camera': (500, 1500)
    }

    countries = ['US', 'UK', 'Canada', 'Australia', 'Germany', 'France']
    payment_methods = ['Credit Card', 'PayPal', 'Debit Card', 'Bank Transfer']

    data = []
    base_date = datetime.now() - timedelta(days=365)

    for i in range(n_records):
        product = np.random.choice(list(products.keys()))
        price_range = products[product]

        # Generate correlated data (higher price � higher quantity)
        base_price = np.random.uniform(price_range[0], price_range[1])
        quantity = np.random.poisson(2) + 1  # Most orders are 1-3 items

        # Add some seasonality (more sales in Nov-Dec)
        days_offset = np.random.randint(0, 365)
        order_date = base_date + timedelta(days=days_offset)

        # Introduce some missing values (realistic data quality issue)
        age = np.random.randint(18, 70) if np.random.random() > 0.05 else None
        rating = np.random.randint(1, 6) if np.random.random() > 0.1 else None

        data.append({
            'order_id': f'ORD{i:05d}',
            'customer_id': f'CUST{np.random.randint(1, 200):04d}',
            'product': product,
            'category': 'Electronics',
            'quantity': quantity,
            'unit_price': round(base_price, 2),
            'total_amount': round(base_price * quantity, 2),
            'country': np.random.choice(countries),
            'payment_method': np.random.choice(payment_methods),
            'customer_age': age,
            'rating': rating,
            'order_date': order_date,
            'is_weekend': order_date.weekday() >= 5
        })

    return pd.DataFrame(data)


# ============================================================================
# STEP 1: DATA COLLECTION & UNDERSTANDING
# ============================================================================
# First step in EDA: Load data and understand its structure
# ============================================================================

def step1_data_understanding(df):
    """
    Step 1: Understand your dataset structure
    - What data do we have?
    - What are the data types?
    - How much data is there?
    """
    print("=" * 80)
    print("STEP 1: DATA UNDERSTANDING".center(80))
    print("=" * 80)

    print("\n[1.1] Dataset Shape")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\n[1.2] First Few Records")
    print(df.head())

    print("\n[1.3] Data Types")
    print(df.dtypes)

    print("\n[1.4] Dataset Info")
    print(df.info())

    print("\n[1.5] Memory Usage")
    print(f"Total memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")


# ============================================================================
# STEP 2: DATA QUALITY ASSESSMENT
# ============================================================================
# Identify missing values, duplicates, and data quality issues
# ============================================================================

def step2_data_quality(df):
    """
    Step 2: Assess data quality
    - Missing values
    - Duplicate records
    - Data inconsistencies
    """
    print("\n" + "=" * 80)
    print("STEP 2: DATA QUALITY ASSESSMENT".center(80))
    print("=" * 80)

    print("\n[2.1] Missing Values Analysis")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    }).sort_values('Missing_Count', ascending=False)
    print(missing_df[missing_df['Missing_Count'] > 0])

    print("\n[2.2] Duplicate Records")
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")

    print("\n[2.3] Data Quality Summary")
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    print(f"Data Completeness: {((total_cells - missing_cells) / total_cells * 100):.2f}%")


# ============================================================================
# STEP 3: UNIVARIATE ANALYSIS
# ============================================================================
# Analyze ONE variable at a time
# - Numerical: mean, median, distribution, outliers
# - Categorical: frequency, mode
# ============================================================================

def step3_univariate_analysis(df):
    """
    Step 3: Univariate Analysis - Analyze each variable individually
    """
    print("\n" + "=" * 80)
    print("STEP 3: UNIVARIATE ANALYSIS".center(80))
    print("=" * 80)

    # ========================================================================
    # 3.1 Numerical Variables Analysis
    # ========================================================================
    print("\n[3.1] Numerical Variables - Descriptive Statistics")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numerical_cols].describe())

    print("\n[3.2] Distribution Analysis")
    print("\nTotal Amount Statistics:")
    print(f"  Mean: ${df['total_amount'].mean():,.2f}")
    print(f"  Median: ${df['total_amount'].median():,.2f}")
    print(f"  Std Dev: ${df['total_amount'].std():,.2f}")
    print(f"  Min: ${df['total_amount'].min():,.2f}")
    print(f"  Max: ${df['total_amount'].max():,.2f}")

    # ========================================================================
    # 3.3 Categorical Variables Analysis
    # ========================================================================
    print("\n[3.3] Categorical Variables - Frequency Distribution")

    print("\nProduct Distribution:")
    print(df['product'].value_counts())

    print("\nCountry Distribution:")
    print(df['country'].value_counts())

    print("\nPayment Method Distribution:")
    print(df['payment_method'].value_counts())

    # ========================================================================
    # 3.4 Visualization (if running in interactive mode)
    # ========================================================================
    print("\n[3.4] Creating Univariate Visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Histogram for numerical variable
    axes[0, 0].hist(df['total_amount'].dropna(), bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Total Amount')
    axes[0, 0].set_xlabel('Total Amount ($)')
    axes[0, 0].set_ylabel('Frequency')

    # Box plot for outlier detection
    axes[0, 1].boxplot(df['total_amount'].dropna())
    axes[0, 1].set_title('Total Amount - Box Plot (Outlier Detection)')
    axes[0, 1].set_ylabel('Total Amount ($)')

    # Bar chart for categorical variable
    product_counts = df['product'].value_counts()
    axes[1, 0].bar(product_counts.index, product_counts.values, color='lightcoral')
    axes[1, 0].set_title('Product Distribution')
    axes[1, 0].set_xlabel('Product')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Pie chart for proportions
    country_counts = df['country'].value_counts()
    axes[1, 1].pie(country_counts.values, labels=country_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Sales by Country')

    plt.tight_layout()
    plt.savefig('images/univariate_analysis.png', dpi=300, bbox_inches='tight')
    print(" Saved: images/univariate_analysis.png")
    plt.close()


# ============================================================================
# STEP 4: BIVARIATE ANALYSIS
# ============================================================================
# Analyze relationships between TWO variables
# - Numerical vs Numerical: correlation, scatter plots
# - Categorical vs Numerical: group comparisons
# - Categorical vs Categorical: cross-tabulation
# ============================================================================

def step4_bivariate_analysis(df):
    """
    Step 4: Bivariate Analysis - Analyze relationships between two variables
    """
    print("\n" + "=" * 80)
    print("STEP 4: BIVARIATE ANALYSIS".center(80))
    print("=" * 80)

    # ========================================================================
    # 4.1 Numerical vs Numerical: Correlation Analysis
    # ========================================================================
    print("\n[4.1] Correlation Analysis (Numerical Variables)")
    numerical_cols = ['quantity', 'unit_price', 'total_amount', 'customer_age']
    correlation_matrix = df[numerical_cols].corr()
    print(correlation_matrix)

    print("\n[4.2] Key Correlations:")
    print(f"Quantity vs Total Amount: {df['quantity'].corr(df['total_amount']):.3f}")
    print(f"Unit Price vs Total Amount: {df['unit_price'].corr(df['total_amount']):.3f}")

    # ========================================================================
    # 4.3 Categorical vs Numerical: Group Comparisons
    # ========================================================================
    print("\n[4.3] Group Comparisons (Categorical vs Numerical)")

    print("\nAverage Total Amount by Product:")
    print(df.groupby('product')['total_amount'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False))

    print("\nAverage Total Amount by Country:")
    print(df.groupby('country')['total_amount'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False))

    # ========================================================================
    # 4.4 Categorical vs Categorical: Cross-tabulation
    # ========================================================================
    print("\n[4.4] Cross-tabulation (Categorical vs Categorical)")

    print("\nProduct vs Country Cross-tab:")
    crosstab = pd.crosstab(df['product'], df['country'])
    print(crosstab)

    # ========================================================================
    # 4.5 Visualization
    # ========================================================================
    print("\n[4.5] Creating Bivariate Visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Scatter plot: Numerical vs Numerical
    axes[0, 0].scatter(df['unit_price'], df['total_amount'], alpha=0.5, color='blue')
    axes[0, 0].set_title('Unit Price vs Total Amount')
    axes[0, 0].set_xlabel('Unit Price ($)')
    axes[0, 0].set_ylabel('Total Amount ($)')

    # Correlation heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                ax=axes[0, 1], fmt='.2f', square=True)
    axes[0, 1].set_title('Correlation Heatmap')

    # Box plot: Categorical vs Numerical
    df.boxplot(column='total_amount', by='product', ax=axes[1, 0])
    axes[1, 0].set_title('Total Amount by Product')
    axes[1, 0].set_xlabel('Product')
    axes[1, 0].set_ylabel('Total Amount ($)')
    plt.sca(axes[1, 0])
    plt.xticks(rotation=45)

    # Grouped bar chart
    product_country = df.groupby(['product', 'country'])['total_amount'].sum().unstack()
    product_country.plot(kind='bar', ax=axes[1, 1], width=0.8)
    axes[1, 1].set_title('Total Sales by Product and Country')
    axes[1, 1].set_xlabel('Product')
    axes[1, 1].set_ylabel('Total Sales ($)')
    axes[1, 1].legend(title='Country', bbox_to_anchor=(1.05, 1))
    plt.sca(axes[1, 1])
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('images/bivariate_analysis.png', dpi=300, bbox_inches='tight')
    print(" Saved: images/bivariate_analysis.png")
    plt.close()


# ============================================================================
# STEP 5: MULTIVARIATE ANALYSIS
# ============================================================================
# Analyze relationships among THREE or more variables
# - Identify complex patterns and interactions
# ============================================================================

def step5_multivariate_analysis(df):
    """
    Step 5: Multivariate Analysis - Analyze multiple variables together
    """
    print("\n" + "=" * 80)
    print("STEP 5: MULTIVARIATE ANALYSIS".center(80))
    print("=" * 80)

    # ========================================================================
    # 5.1 Multiple Group Analysis
    # ========================================================================
    print("\n[5.1] Multi-dimensional Group Analysis")

    print("\nAverage Sales by Product, Country, and Payment Method:")
    multi_group = df.groupby(['product', 'country', 'payment_method'])['total_amount'].agg(['mean', 'count'])
    print(multi_group.sort_values('mean', ascending=False).head(10))

    # ========================================================================
    # 5.2 Time-based Analysis
    # ========================================================================
    print("\n[5.2] Time-based Patterns")

    df['month'] = df['order_date'].dt.month
    df['day_of_week'] = df['order_date'].dt.day_name()

    print("\nSales by Month:")
    monthly_sales = df.groupby('month')['total_amount'].agg(['sum', 'mean', 'count'])
    print(monthly_sales)

    print("\nWeekend vs Weekday Sales:")
    print(df.groupby('is_weekend')['total_amount'].agg(['sum', 'mean', 'count']))

    # ========================================================================
    # 5.3 Visualization
    # ========================================================================
    print("\n[5.3] Creating Multivariate Visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Time series
    monthly_totals = df.groupby('month')['total_amount'].sum()
    axes[0, 0].plot(monthly_totals.index, monthly_totals.values, marker='o', linewidth=2)
    axes[0, 0].set_title('Monthly Sales Trend')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Total Sales ($)')
    axes[0, 0].grid(True)

    # Scatter plot with hue (3 variables)
    for product in df['product'].unique():
        product_data = df[df['product'] == product]
        axes[0, 1].scatter(product_data['unit_price'], product_data['quantity'],
                          label=product, alpha=0.6, s=50)
    axes[0, 1].set_title('Unit Price vs Quantity by Product')
    axes[0, 1].set_xlabel('Unit Price ($)')
    axes[0, 1].set_ylabel('Quantity')
    axes[0, 1].legend()

    # Heatmap: Product vs Country
    product_country_pivot = df.pivot_table(values='total_amount',
                                           index='product',
                                           columns='country',
                                           aggfunc='sum')
    sns.heatmap(product_country_pivot, annot=True, fmt='.0f', cmap='YlGnBu', ax=axes[1, 0])
    axes[1, 0].set_title('Total Sales: Product vs Country')

    # Multiple box plots
    df.boxplot(column='total_amount', by=['product', 'is_weekend'], ax=axes[1, 1])
    axes[1, 1].set_title('Sales Distribution: Product vs Weekend')
    axes[1, 1].set_xlabel('Product, Weekend')
    axes[1, 1].set_ylabel('Total Amount ($)')
    plt.sca(axes[1, 1])
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('images/multivariate_analysis.png', dpi=300, bbox_inches='tight')
    print(" Saved: images/multivariate_analysis.png")
    plt.close()


# ============================================================================
# STEP 6: KEY INSIGHTS & PATTERNS
# ============================================================================
# Summarize findings and actionable insights
# ============================================================================

def step6_insights(df):
    """
    Step 6: Extract Key Insights from EDA
    """
    print("\n" + "=" * 80)
    print("STEP 6: KEY INSIGHTS & PATTERNS".center(80))
    print("=" * 80)

    print("\n[6.1] Business Insights")

    # Top performing products
    top_product = df.groupby('product')['total_amount'].sum().sort_values(ascending=False).head(1)
    print(f"\n Top Product: {top_product.index[0]} (${top_product.values[0]:,.2f} in sales)")

    # Best market
    top_country = df.groupby('country')['total_amount'].sum().sort_values(ascending=False).head(1)
    print(f" Best Market: {top_country.index[0]} (${top_country.values[0]:,.2f} in sales)")

    # Average order value
    avg_order = df['total_amount'].mean()
    print(f" Average Order Value: ${avg_order:,.2f}")

    # Customer demographics
    avg_age = df['customer_age'].mean()
    print(f" Average Customer Age: {avg_age:.1f} years")

    # Payment preferences
    top_payment = df['payment_method'].value_counts().head(1)
    print(f" Most Popular Payment: {top_payment.index[0]} ({top_payment.values[0]} orders)")

    # Weekend effect
    weekend_avg = df[df['is_weekend']]['total_amount'].mean()
    weekday_avg = df[~df['is_weekend']]['total_amount'].mean()
    diff_pct = ((weekend_avg - weekday_avg) / weekday_avg) * 100
    print(f" Weekend vs Weekday: {diff_pct:+.1f}% difference")

    print("\n[6.2] Data Quality Insights")
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    print(f" Overall Data Completeness: {100 - missing_pct:.1f}%")

    print("\n[6.3] Recommendations")
    print("  1. Focus marketing efforts on top-performing products")
    print("  2. Expand presence in high-revenue markets")
    print("  3. Investigate weekend sales patterns for optimization")
    print("  4. Address missing data in customer_age and rating fields")
    print("  5. Consider targeted promotions for underperforming products")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" EXPLORATORY DATA ANALYSIS (EDA) DEMONSTRATION ".center(80))
    print("=" * 80)
    print("\nWhat is EDA?")
    print("EDA is the process of analyzing datasets to summarize their main")
    print("characteristics, often using visual methods. It helps to:")
    print("  • Understand data structure and quality")
    print("  • Identify patterns, trends, and anomalies")
    print("  • Detect outliers and missing values")
    print("  • Test assumptions and hypotheses")
    print("  • Guide feature engineering for machine learning")

    # Generate sample data
    print("\n[SETUP] Generating sample e-commerce dataset...")
    df = generate_ecommerce_data(500)
    print(f" Generated {len(df)} records")

    # Execute EDA steps
    step1_data_understanding(df)
    step2_data_quality(df)
    step3_univariate_analysis(df)
    step4_bivariate_analysis(df)
    step5_multivariate_analysis(df)
    step6_insights(df)

    # Save processed data
    df.to_csv('eda_sample_data.csv', index=False)
    print("\n" + "=" * 80)
    print(" EDA COMPLETE ".center(80))
    print("=" * 80)
    print("\nOutput Files Generated:")
    print("  • eda_sample_data.csv - Sample dataset")
    print("  • images/univariate_analysis.png - Single variable visualizations")
    print("  • images/bivariate_analysis.png - Two variable relationships")
    print("  • images/multivariate_analysis.png - Multiple variable patterns")

    print("\nKey Takeaways:")
    print("  • Always start with data understanding and quality checks")
    print("  • Use univariate analysis to understand individual variables")
    print("  • Use bivariate analysis to find relationships between pairs")
    print("  • Use multivariate analysis for complex pattern discovery")
    print("  • Visualizations are crucial for identifying patterns")

    print("\nNext Steps:")
    print("  • Learn advanced visualization with Seaborn and Plotly")
    print("  • Explore automated EDA tools (ydata-profiling)")
    print("  • Practice with real-world datasets (Kaggle, UCI)")
    print("  • Study feature engineering techniques")
