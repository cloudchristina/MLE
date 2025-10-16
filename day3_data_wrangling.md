# Data Wrangling: A Beginner's Guide

Data wrangling (also known as data munging) is the process of transforming raw data into a clean, structured format that's ready for analysis. Think of it as "preparing ingredients before cooking" - you need to clean, chop, and organize everything before you can create a great dish.

## Table of Contents
- [What is Data Wrangling?](#what-is-data-wrangling)
- [Why is Data Wrangling Important?](#why-is-data-wrangling-important)
- [Data Wrangling Process](#data-wrangling-process)
- [Common Data Wrangling Tasks](#common-data-wrangling-tasks)
- [Tools and Techniques](#tools-and-techniques)
- [Best Practices](#best-practices)
- [Getting Started](#getting-started)

## What is Data Wrangling?

Data wrangling is the process of:
- **Cleaning** messy, inconsistent data
- **Transforming** data into the right format
- **Enriching** data with additional information
- **Validating** data quality and accuracy
- **Preparing** data for analysis or machine learning

```mermaid
flowchart LR
    Raw[Raw Data<br/>Messy, Incomplete] --> Wrangle[Data Wrangling<br/>Clean & Transform]
    Wrangle --> Clean[Clean Data<br/>Ready for Analysis]

    Clean --> Analysis[Data Analysis]
    Clean --> ML[Machine Learning]
    Clean --> Viz[Visualization]

    style Raw fill:#ffebee
    style Wrangle fill:#fff4e1
    style Clean fill:#e8f5e9
    style Analysis fill:#e3f2fd
    style ML fill:#e3f2fd
    style Viz fill:#e3f2fd
```

## Why is Data Wrangling Important?

### The Reality of Real-World Data

```mermaid
graph TB
    RealWorld[Real-World Data] --> Issues[Common Issues]

    Issues --> Missing[Missing Values<br/>20-30% incomplete]
    Issues --> Duplicates[Duplicates<br/>5-10% redundant]
    Issues --> Inconsistent[Inconsistencies<br/>Different formats]
    Issues --> Errors[Errors<br/>Invalid values]
    Issues --> Outliers[Outliers<br/>Extreme values]

    Missing --> Impact[Impact on Analysis]
    Duplicates --> Impact
    Inconsistent --> Impact
    Errors --> Impact
    Outliers --> Impact

    Impact --> Wrong[Wrong Insights]
    Impact --> Biased[Biased Models]
    Impact --> Failed[Failed Analysis]

    style RealWorld fill:#ffebee
    style Issues fill:#fff4e1
    style Impact fill:#fce4ec
    style Wrong fill:#ffcdd2
    style Biased fill:#ffcdd2
    style Failed fill:#ffcdd2
```

**Key Statistics:**
- Data scientists spend **60-80%** of their time on data wrangling
- Poor data quality costs organizations an average of **$15 million** per year
- **Only 3%** of data meets basic quality standards without wrangling

**Benefits of Data Wrangling:**
1. **Accuracy**: Clean data leads to accurate insights
2. **Efficiency**: Structured data speeds up analysis
3. **Reliability**: Validated data ensures trustworthy results
4. **Compliance**: Proper data handling meets regulations
5. **Decision-Making**: Quality data enables better decisions

## Data Wrangling Process

### The 4-Step Framework

```mermaid
flowchart TD
    Start([Raw Data]) --> Step1[1. Discovery<br/>Explore & Understand]
    Step1 --> Step2[2. Cleaning<br/>Fix Errors & Missing Data]
    Step2 --> Step3[3. Transformation<br/>Structure & Enrich]
    Step3 --> Step4[4. Validation<br/>Verify Quality]
    Step4 --> Decision{Data<br/>Ready?}
    Decision -->|No| Step2
    Decision -->|Yes| End([Clean Data])

    style Step1 fill:#e3f2fd
    style Step2 fill:#ffebee
    style Step3 fill:#fff4e1
    style Step4 fill:#e8f5e9
    style End fill:#c8e6c9
```

### Step 1: Discovery

**Objective**: Understand what you're working with

```mermaid
graph LR
    Discovery[Discovery Phase] --> Questions[Key Questions]

    Questions --> Q1[What data do I have?]
    Questions --> Q2[Where did it come from?]
    Questions --> Q3[What's the structure?]
    Questions --> Q4[What are the issues?]

    Q1 --> Actions[Actions]
    Q2 --> Actions
    Q3 --> Actions
    Q4 --> Actions

    Actions --> A1[Load & Preview]
    Actions --> A2[Check Data Types]
    Actions --> A3[Identify Patterns]
    Actions --> A4[Document Findings]

    style Discovery fill:#e3f2fd
    style Questions fill:#bbdefb
    style Actions fill:#90caf9
```

**Tasks:**
- Load data from various sources
- Examine first/last records
- Check dimensions (rows � columns)
- Identify data types
- Look for obvious issues

**Example Insights:**
- "This dataset has 10,000 rows and 15 columns"
- "The 'date' column is stored as text, not dates"
- "About 20% of 'age' values are missing"

### Step 2: Cleaning

**Objective**: Fix errors and handle missing data

```mermaid
flowchart TD
    Cleaning[Data Cleaning] --> Task1[Handle Missing<br/>Values]
    Cleaning --> Task2[Remove<br/>Duplicates]
    Cleaning --> Task3[Fix Data<br/>Types]
    Cleaning --> Task4[Standardize<br/>Formats]
    Cleaning --> Task5[Handle<br/>Outliers]

    Task1 --> Method1[Delete, Impute,<br/>or Flag]
    Task2 --> Method2[Drop exact<br/>duplicates]
    Task3 --> Method3[Convert to<br/>correct type]
    Task4 --> Method4[Consistent<br/>formatting]
    Task5 --> Method5[Cap, remove,<br/>or investigate]

    style Cleaning fill:#ffebee
    style Task1 fill:#ffcdd2
    style Task2 fill:#ffcdd2
    style Task3 fill:#ffcdd2
    style Task4 fill:#ffcdd2
    style Task5 fill:#ffcdd2
```

**Common Cleaning Tasks:**

| Issue | Solution | Example |
|-------|----------|---------|
| Missing Values | Impute with mean/median/mode | Fill age with median age |
| Duplicates | Remove identical rows | Keep first occurrence |
| Wrong Types | Convert to correct type | "123" � 123 |
| Inconsistent Format | Standardize | "USA", "US", "United States" � "US" |
| Outliers | Cap or remove | Age = 999 � remove |

### Step 3: Transformation

**Objective**: Structure and enrich data for analysis

```mermaid
flowchart LR
    Transform[Transformation] --> Struct[Restructure]
    Transform --> Derive[Derive Features]
    Transform --> Merge[Merge Data]
    Transform --> Aggregate[Aggregate]

    Struct --> S1[Pivot/Unpivot<br/>Wide � Long]
    Derive --> D1[Create new columns<br/>from existing]
    Merge --> M1[Join multiple<br/>datasets]
    Aggregate --> A1[Group & summarize<br/>by categories]

    style Transform fill:#fff4e1
    style Struct fill:#ffe082
    style Derive fill:#ffe082
    style Merge fill:#ffe082
    style Aggregate fill:#ffe082
```

**Transformation Types:**

**1. Feature Engineering**
- Create age groups from age (18-25, 26-35, etc.)
- Extract month/day from dates
- Calculate ratios (revenue/cost)

**2. Data Reshaping**
- Pivot: Long format � Wide format
- Melt: Wide format � Long format
- Stack/Unstack operations

**3. Data Merging**
- Join datasets on common keys
- Concatenate similar datasets
- Combine horizontal/vertical

**4. Aggregation**
- Group by categories (product, region)
- Calculate summaries (sum, mean, count)
- Time-based aggregation (daily � monthly)

### Step 4: Validation

**Objective**: Verify data quality and correctness

```mermaid
flowchart TD
    Validation[Validation Phase] --> Check1[Completeness<br/>Check]
    Validation --> Check2[Accuracy<br/>Check]
    Validation --> Check3[Consistency<br/>Check]
    Validation --> Check4[Business Rules<br/>Check]

    Check1 --> Test1{All required<br/>fields present?}
    Check2 --> Test2{Values in<br/>valid range?}
    Check3 --> Test3{Data follows<br/>standards?}
    Check4 --> Test4{Meets business<br/>logic?}

    Test1 -->|Yes| Pass1[]
    Test1 -->|No| Fail1[ Fix]
    Test2 -->|Yes| Pass2[]
    Test2 -->|No| Fail2[ Fix]
    Test3 -->|Yes| Pass3[]
    Test3 -->|No| Fail3[ Fix]
    Test4 -->|Yes| Pass4[]
    Test4 -->|No| Fail4[ Fix]

    Fail1 --> Check1
    Fail2 --> Check2
    Fail3 --> Check3
    Fail4 --> Check4

    style Validation fill:#e8f5e9
    style Pass1 fill:#c8e6c9
    style Pass2 fill:#c8e6c9
    style Pass3 fill:#c8e6c9
    style Pass4 fill:#c8e6c9
    style Fail1 fill:#ffcdd2
    style Fail2 fill:#ffcdd2
    style Fail3 fill:#ffcdd2
    style Fail4 fill:#ffcdd2
```

**Validation Checks:**

1. **Completeness**: No missing critical values
2. **Accuracy**: Values within expected ranges
3. **Consistency**: Formats match standards
4. **Uniqueness**: No unexpected duplicates
5. **Integrity**: Relationships preserved

## Common Data Wrangling Tasks

### 1. Handling Missing Values

```mermaid
graph TD
    Missing[Missing Values] --> Strategy{Strategy?}

    Strategy -->|Delete| Delete[Remove rows<br/>or columns]
    Strategy -->|Impute| Impute[Fill with<br/>values]
    Strategy -->|Flag| Flag[Keep & mark<br/>as missing]

    Delete --> When1[When: < 5% missing<br/>Not critical]
    Impute --> When2[When: Can estimate<br/>from other data]
    Flag --> When3[When: Missingness<br/>is meaningful]

    Impute --> Methods[Methods]
    Methods --> Mean[Mean/Median<br/>for numbers]
    Methods --> Mode[Mode for<br/>categories]
    Methods --> Forward[Forward/Backward<br/>fill for time series]
    Methods --> ML[ML-based<br/>prediction]

    style Missing fill:#ffebee
    style Delete fill:#ffcdd2
    style Impute fill:#fff4e1
    style Flag fill:#e3f2fd
```

**Decision Guide:**
- **< 5% missing**: Usually safe to delete
- **5-40% missing**: Consider imputation
- **> 40% missing**: Might need to drop column or collect more data

### 2. Removing Duplicates

```mermaid
flowchart LR
    Data[Dataset] --> Check[Check for<br/>Duplicates]
    Check --> Type{Duplicate<br/>Type?}

    Type -->|Exact| Exact[Identical rows<br/>Keep first]
    Type -->|Partial| Partial[Same key,<br/>different values]
    Type -->|Fuzzy| Fuzzy[Similar but<br/>not exact]

    Exact --> Action1[df.drop_duplicates]
    Partial --> Action2[Merge & resolve<br/>conflicts]
    Fuzzy --> Action3[String matching<br/>algorithms]

    style Data fill:#e3f2fd
    style Check fill:#bbdefb
    style Exact fill:#ffcdd2
    style Partial fill:#fff4e1
    style Fuzzy fill:#fce4ec
```

### 3. Data Type Conversion

```mermaid
graph LR
    Types[Common Conversions] --> Numeric[Text � Number]
    Types --> Date[Text � Date]
    Types --> Category[Text � Category]
    Types --> Boolean[Text � Boolean]

    Numeric --> N1["'123' � 123"]
    Date --> D1["'2024-01-01' � datetime"]
    Category --> C1["'red' � category"]
    Boolean --> B1["'yes' � True"]

    N1 --> Issues1[Handle: '$1,234.56'<br/>Remove symbols]
    D1 --> Issues2[Handle: Multiple<br/>date formats]
    C1 --> Issues3[Handle: Reduce<br/>memory usage]
    B1 --> Issues4[Handle: Various<br/>representations]

    style Types fill:#e3f2fd
    style Numeric fill:#bbdefb
    style Date fill:#bbdefb
    style Category fill:#bbdefb
    style Boolean fill:#bbdefb
```

### 4. Standardization & Normalization

```mermaid
flowchart TD
    Standard[Standardization] --> Text[Text Data]
    Standard --> Numeric[Numeric Data]

    Text --> T1[Case conversion<br/>UPPER/lower/Title]
    Text --> T2[Remove whitespace<br/>trim/strip]
    Text --> T3[Fix encoding<br/>UTF-8, ASCII]

    Numeric --> N1[Scaling<br/>0-1 range]
    Numeric --> N2[Normalization<br/>Mean=0, Std=1]
    Numeric --> N3[Rounding<br/>Decimal places]

    T1 --> Example1["'USA', 'usa' � 'USA'"]
    T2 --> Example2["'  text  ' � 'text'"]
    N1 --> Example3["[0,100] � [0,1]"]

    style Standard fill:#fff4e1
    style Text fill:#ffe082
    style Numeric fill:#ffe082
```

### 5. Feature Engineering

```mermaid
graph TD
    Feature[Feature Engineering] --> Create[Create New<br/>Features]

    Create --> Time[Time-based<br/>Features]
    Create --> Math[Mathematical<br/>Operations]
    Create --> Encode[Encoding<br/>Categorical]
    Create --> Bin[Binning<br/>Continuous]

    Time --> T1[Year, Month, Day<br/>Weekday, Hour]
    Math --> M1[Ratios, Differences<br/>Products, Logs]
    Encode --> E1[One-hot, Label<br/>Target encoding]
    Bin --> B1[Age groups<br/>Price tiers]

    style Feature fill:#e8f5e9
    style Create fill:#c8e6c9
    style Time fill:#a5d6a7
    style Math fill:#a5d6a7
    style Encode fill:#a5d6a7
    style Bin fill:#a5d6a7
```

## Tools and Techniques

### Python Libraries for Data Wrangling

```mermaid
graph LR
    Python[Python<br/>Data Wrangling] --> Core[Core Libraries]
    Python --> Advanced[Advanced Tools]

    Core --> Pandas[Pandas<br/>Primary tool]
    Core --> NumPy[NumPy<br/>Numerical ops]

    Advanced --> Dask[Dask<br/>Large datasets]
    Advanced --> Vaex[Vaex<br/>Out-of-memory]
    Advanced --> Polars[Polars<br/>Fast & efficient]

    Pandas --> P1[read_csv, fillna<br/>drop_duplicates]
    NumPy --> N1[Arrays, math<br/>operations]
    Dask --> D1[Parallel processing<br/>Big data]

    style Python fill:#e8f5e9
    style Core fill:#c8e6c9
    style Advanced fill:#a5d6a7
```

### Essential Pandas Operations

| Task | Pandas Method | Example |
|------|--------------|---------|
| **Load Data** | `pd.read_csv()` | `df = pd.read_csv('data.csv')` |
| **Preview** | `.head()`, `.tail()` | `df.head(10)` |
| **Info** | `.info()`, `.describe()` | `df.info()` |
| **Missing** | `.isnull()`, `.fillna()` | `df.fillna(0)` |
| **Duplicates** | `.drop_duplicates()` | `df.drop_duplicates()` |
| **Filter** | Boolean indexing | `df[df['age'] > 18]` |
| **Select** | `.loc[]`, `.iloc[]` | `df.loc[:, ['name', 'age']]` |
| **Group** | `.groupby()` | `df.groupby('category').mean()` |
| **Merge** | `.merge()`, `.join()` | `pd.merge(df1, df2, on='id')` |
| **Reshape** | `.pivot()`, `.melt()` | `df.pivot(index='date', columns='product')` |

### Data Wrangling Workflow

```mermaid
flowchart TD
    Start([Start]) --> Load[Load Data<br/>pd.read_csv]
    Load --> Explore[Explore<br/>.info, .describe]
    Explore --> Missing[Handle Missing<br/>.fillna, .dropna]
    Missing --> Duplicates[Remove Duplicates<br/>.drop_duplicates]
    Duplicates --> Types[Fix Data Types<br/>.astype]
    Types --> Transform[Transform<br/>Feature engineering]
    Transform --> Validate[Validate<br/>Quality checks]
    Validate --> Save[Save Clean Data<br/>.to_csv]
    Save --> End([End])

    style Load fill:#e3f2fd
    style Explore fill:#e3f2fd
    style Missing fill:#ffebee
    style Duplicates fill:#ffebee
    style Types fill:#fff4e1
    style Transform fill:#fff4e1
    style Validate fill:#e8f5e9
    style Save fill:#c8e6c9
```

## Best Practices

### 1. Document Your Process

```mermaid
graph LR
    Document[Documentation] --> What[What was done?]
    Document --> Why[Why was it done?]
    Document --> How[How was it done?]

    What --> W1[Log transformations]
    Why --> W2[Explain decisions]
    How --> H1[Code comments]

    W1 --> Output[Clear audit trail]
    W2 --> Output
    H1 --> Output

    style Document fill:#e3f2fd
    style Output fill:#c8e6c9
```

**Best Practices:**
- Comment your code explaining WHY, not just WHAT
- Keep a log of data quality issues found
- Document assumptions and decisions
- Version control your scripts

### 2. Validate at Every Step

```mermaid
flowchart LR
    Step1[Cleaning Step] --> Check1[Validate]
    Check1 -->|Pass| Step2[Transform Step]
    Check1 -->|Fail| Fix1[Fix Issues]
    Fix1 --> Step1

    Step2 --> Check2[Validate]
    Check2 -->|Pass| Step3[Next Step]
    Check2 -->|Fail| Fix2[Fix Issues]
    Fix2 --> Step2

    style Check1 fill:#e8f5e9
    style Check2 fill:#e8f5e9
    style Fix1 fill:#ffcdd2
    style Fix2 fill:#ffcdd2
```

**Validation Checklist:**
-  Check data shape before/after
-  Verify no unexpected nulls introduced
-  Confirm data types are correct
-  Test with sample calculations
-  Compare totals/counts

### 3. Handle Edge Cases

**Common Edge Cases:**
- Empty strings vs. null values
- Leading/trailing whitespace
- Different date formats
- Special characters in text
- Numeric values as strings
- Zero vs. null vs. missing

### 4. Keep Raw Data Unchanged

```mermaid
flowchart LR
    Raw[Raw Data<br/>Immutable] --> Copy[Create Copy<br/>df.copy]
    Copy --> Wrangle[Wrangle Copy]
    Wrangle --> Clean[Clean Data]

    Raw -.->|Always preserve| Archive[Archived<br/>for reference]

    style Raw fill:#e3f2fd
    style Archive fill:#e3f2fd
    style Clean fill:#e8f5e9
```

**Why?**
- Allows reprocessing if needed
- Enables comparison with original
- Maintains data lineage
- Supports auditability

### 5. Automate Repetitive Tasks

```python
# Create reusable functions
def clean_dataframe(df):
    """Standard cleaning pipeline"""
    df = df.copy()
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = fix_data_types(df)
    df = standardize_formats(df)
    return df
```

## Getting Started

### Prerequisites

```bash
# Install required libraries
pip install pandas numpy matplotlib seaborn

# Optional: For larger datasets
pip install dask vaex polars
```

### Running the Demo

```bash
# Run the comprehensive data wrangling demo
python day3_data_wrangling.py
```

**What the demo includes:**
- Sample messy dataset generation
- All 4 steps of data wrangling process
- Common wrangling tasks demonstrated
- Before/after comparisons
- Quality metrics reporting

### Expected Output

**Console Output:**
- Data discovery insights
- Cleaning operations performed
- Transformation summaries
- Validation results
- Quality improvement metrics

**Generated Files:**
- `raw_data.csv` - Original messy data
- `clean_data.csv` - Wrangled clean data
- `wrangling_report.txt` - Process documentation

## Real-World Example: E-Commerce Data

### Before Wrangling

```
| order_id | customer | product  | price    | date       | quantity |
|----------|----------|----------|----------|------------|----------|
| 1        | john doe | laptop   | $1,200   | 01/15/2024 | 1        |
| 2        | JANE DOE | Phone    | 800.00   | 2024-01-16 | 2        |
| 1        | john doe | laptop   | $1,200   | 01/15/2024 | 1        |
| 3        |          | tablet   | NULL     | 15-01-2024 |          |
```

**Issues:**
- Inconsistent name formatting (case, spaces)
- Multiple date formats
- Price has symbols and inconsistent format
- Missing values (customer, quantity)
- Duplicate rows
- Null values in critical fields

### After Wrangling

```
| order_id | customer | product | price  | date       | quantity | revenue |
|----------|----------|---------|--------|------------|----------|---------|
| 1        | John Doe | Laptop  | 1200.0 | 2024-01-15 | 1        | 1200.0  |
| 2        | Jane Doe | Phone   | 800.0  | 2024-01-16 | 2        | 1600.0  |
```

**Improvements:**
-  Standardized name format (Title Case)
-  Unified date format (YYYY-MM-DD)
-  Clean numeric prices
-  Removed duplicates
-  Handled missing values
-  Added derived column (revenue)

## Key Takeaways

1. **Data wrangling is essential** - 60-80% of data science time
2. **Follow a systematic process** - Discovery � Clean � Transform � Validate
3. **Document everything** - Maintain audit trail
4. **Validate continuously** - Check at every step
5. **Automate when possible** - Reusable functions for common tasks
6. **Keep raw data safe** - Never modify originals
7. **Think about edge cases** - Handle unexpected inputs


## Further Learning

### Recommended Resources

- [DataCamp: What is Data Wrangling?](https://www.datacamp.com/blog/what-is-data-wrangling)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Real Python - Pandas Tutorials](https://realpython.com/learning-paths/pandas-data-science/)

### Practice Datasets

- **Kaggle**: Real-world messy datasets
- **Data.gov**: Government open data
- **UCI ML Repository**: Classic datasets with quality issues
- **Your own data**: Practice with real problems

### Next Steps

1. Practice with the provided demo code
2. Work through the exercises
3. Apply to your own datasets
4. Build a data wrangling toolkit
5. Learn data validation frameworks
6. Progress to ETL and data pipelines

---
