"""
Data Ingestion Patterns Demo
===============================
A beginner-friendly demonstration of:
- Batch Ingestion (scheduled processing)
- Streaming Ingestion (real-time processing)
- ETL Pipeline (with quality checks and transformations)

Author: Christina Chen
Learning Resource: https://github.com/cloudchristina/MLE
"""

import pandas as pd
from datetime import datetime, timedelta
import random
import time

# ============================================================================
# GENERATE SAMPLE DATA
# ============================================================================

def generate_sample_data(num_records: int = 100) -> pd.DataFrame:
    """Generate sample e-commerce data"""
    products = ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Watch']
    countries = ['US', 'UK', 'Canada', 'Australia', 'Germany']

    data = []
    base_date = datetime.now() - timedelta(days=30)

    for i in range(num_records):
        data.append({
            'order_id': f'ORD{i:05d}',
            'customer_id': f'CUST{random.randint(1, 50):03d}',
            'product': random.choice(products),
            'quantity': random.randint(1, 5),
            'price': round(random.uniform(50, 2000), 2),
            'country': random.choice(countries),
            'order_date': base_date + timedelta(days=random.randint(0, 30))
        })

    return pd.DataFrame(data)


# ============================================================================
# BATCH INGESTION
# ============================================================================
# Batch ingestion collects and processes data at scheduled intervals.
# Common in traditional ETL workflows for data warehousing.
#
# WHEN TO USE:
# ✓ End-of-day sales reports and financial reconciliation
# ✓ Historical data analysis and business intelligence
# ✓ Data warehouse loading (scheduled jobs)
# ✓ Large-scale data transformations
#
# TOOLS: Apache Airflow, AWS Glue, Azure Data Factory, Talend
#
# PROS: Cost-effective, efficient for large volumes, simpler architecture
# CONS: Higher latency, not suitable for real-time use cases
# ============================================================================

def batch_ingestion_example():
    """
    Batch Ingestion: Process data in scheduled intervals
    Perfect for: Daily reports, historical analysis, periodic updates
    """
    print("=" * 70)
    print("BATCH INGESTION EXAMPLE")
    print("=" * 70)

    # ========================================================================
    # STEP 1: EXTRACT - Retrieve data from source system
    # ========================================================================
    # In batch processing, this typically runs on a schedule (cron job, etc.)
    # Data is pulled in bulk from databases, APIs, files, or data warehouses
    # ========================================================================
    print("\n[STEP 1: EXTRACT]")
    raw_data = generate_sample_data(100)
    print(f"✓ Extracted {len(raw_data)} records from source")
    print(f"\nSample data:\n{raw_data.head(3)}")

    # ========================================================================
    # STEP 2: TRANSFORM - Clean, enrich, and prepare data
    # ========================================================================
    # All records are transformed together, allowing for optimizations like
    # vectorized operations (pandas) that process entire columns at once
    # ========================================================================
    print("\n[STEP 2: TRANSFORM]")

    # TRANSFORMATION #1: Calculate revenue for all records at once
    # Vectorized operation: processes entire column in single operation
    # Much faster than processing one record at a time
    raw_data['revenue'] = raw_data['quantity'] * raw_data['price']

    # TRANSFORMATION #2: Add processing metadata for traceability
    # Useful for debugging, auditing, and tracking when data was processed
    raw_data['processed_date'] = datetime.now()
    # Batch ID helps identify which processing run created this data
    raw_data['batch_id'] = f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # TRANSFORMATION #3: Filter out invalid records
    # Business rule: Only keep orders with positive revenue
    # .copy() prevents SettingWithCopyWarning in pandas
    clean_data = raw_data[raw_data['revenue'] > 0].copy()

    print(f"✓ Transformed data: {len(clean_data)} valid records")
    print(f"✓ Revenue calculated for all orders")

    # ========================================================================
    # STEP 3: LOAD - Write processed data to destination
    # ========================================================================
    # In production, this might write to a database, data warehouse (Snowflake,
    # BigQuery), or data lake (S3). Here we use CSV for simplicity.
    # ========================================================================
    print("\n[STEP 3: LOAD]")
    output_file = 'batch_output.csv'
    clean_data.to_csv(output_file, index=False)
    print(f"✓ Data loaded to {output_file}")

    # ========================================================================
    # BATCH SUMMARY - Aggregate metrics calculated after processing
    # ========================================================================
    # Batch processing enables easy aggregation across entire dataset
    # This is efficient because all data is already in memory
    # ========================================================================
    print("\n[BATCH SUMMARY]")
    print(f"Total Orders: {len(clean_data)}")
    print(f"Total Revenue: ${clean_data['revenue'].sum():,.2f}")
    print(f"Average Order Value: ${clean_data['revenue'].mean():,.2f}")
    print(f"\nRevenue by Country:")
    # groupby() aggregates all records by country - typical batch operation
    print(clean_data.groupby('country')['revenue'].sum().sort_values(ascending=False))

    return clean_data


# ============================================================================
# STREAMING INGESTION
# ============================================================================
# Streaming ingestion processes data continuously as events arrive,
# enabling real-time insights and immediate action.
#
# WHEN TO USE:
# ✓ Fraud detection systems (credit cards, banking)
# ✓ Live dashboards and monitoring
# ✓ IoT sensor data processing
# ✓ Real-time recommendations and personalization
#
# TOOLS: Apache Kafka, AWS Kinesis, Google Cloud Dataflow, Apache Flink
#
# KEY CONCEPTS:
# - Micro-batching: Processing small groups of events for efficiency
# - Windowing: Time-based or count-based data grouping
# - Event-driven architecture: React to data as it arrives
#
# PROS: Low latency (seconds), immediate insights, continuous processing
# CONS: Complex infrastructure, higher costs, requires specialized expertise
# ============================================================================

def streaming_ingestion_example():
    """
    Streaming Ingestion: Process data in real-time as it arrives
    Perfect for: Live dashboards, real-time alerts, IoT data
    """
    print("\n" + "=" * 70)
    print("STREAMING INGESTION EXAMPLE")
    print("=" * 70)

    # ========================================================================
    # STREAMING SETUP - Initialize buffer for micro-batching
    # ========================================================================
    # Buffer: Temporary storage for incoming events before processing
    # Micro-batching: Process small groups for efficiency vs pure event-by-event
    # ========================================================================
    buffer = []           # Holds events until buffer_size is reached
    buffer_size = 10      # Process every 10 events (tunable for latency/throughput)
    total_processed = 0   # Running count of all events processed

    print("\n[STREAMING MODE] Processing events as they arrive...")
    print(f"Buffer size: {buffer_size} events\n")

    # ========================================================================
    # STREAMING LOOP - Simulate continuous event arrival
    # ========================================================================
    # In production, this would be a message queue consumer (Kafka, Kinesis)
    # or webhook receiver processing events from external systems
    # ========================================================================
    for i in range(30):
        # ====================================================================
        # EVENT ARRIVAL - Create new event (simulates user action/sensor data)
        # ====================================================================
        # In real systems, events arrive unpredictably from:
        # - User clicks on website
        # - IoT sensor readings
        # - API calls from mobile apps
        # - Log entries from services
        # ====================================================================
        event = {
            'event_id': i,
            'timestamp': datetime.now().isoformat(),  # When event occurred
            'customer_id': f'CUST{random.randint(1, 50):03d}',
            'event_type': random.choice(['page_view', 'add_to_cart', 'purchase']),
            'product': random.choice(['Laptop', 'Phone', 'Tablet']),
            'value': round(random.uniform(10, 1000), 2)
        }

        # ====================================================================
        # BUFFERING - Add event to micro-batch buffer
        # ====================================================================
        buffer.append(event)
        # Log event arrival with millisecond precision (important for streaming)
        print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Event {i}: {event['event_type']} - ${event['value']:.2f}")

        # ====================================================================
        # MICRO-BATCH PROCESSING - Process when buffer reaches threshold
        # ====================================================================
        # Trade-off: Smaller buffer = lower latency, higher overhead
        #            Larger buffer = higher latency, better throughput
        # ====================================================================
        if len(buffer) >= buffer_size:
            print(f"\n>>> Processing buffer of {len(buffer)} events...")

            # Convert buffer to DataFrame for batch-like operations
            # This is the "micro" in micro-batching - processing small groups
            df = pd.DataFrame(buffer)

            # ==================================================================
            # REAL-TIME AGGREGATIONS - Calculate metrics on current window
            # ==================================================================
            # These calculations happen continuously, providing live insights
            # Unlike batch, we can't see the "whole picture", only current window
            # ==================================================================
            total_value = df['value'].sum()
            purchase_count = len(df[df['event_type'] == 'purchase'])

            print(f"    Total value: ${total_value:.2f}")
            print(f"    Purchases: {purchase_count}")
            print(f"    Events/second: {len(buffer) / 1.0:.1f}")
            print()

            # Track total processed and clear buffer for next micro-batch
            total_processed += len(buffer)
            buffer = []  # Reset buffer - crucial to prevent memory buildup

        # ====================================================================
        # SIMULATE REAL-TIME DELAY - Events arrive over time, not all at once
        # ====================================================================
        # In production, this delay is natural (events arrive when users act)
        # Here we simulate with sleep to demonstrate streaming behavior
        # ====================================================================
        time.sleep(0.1)  # 100ms between events

    # ========================================================================
    # FINAL FLUSH - Process any remaining events in buffer
    # ========================================================================
    # Important: Always process remaining events when stream ends or pauses
    # Otherwise, you lose the last partial batch of data
    # ========================================================================
    if buffer:
        print(f"\n>>> Processing final buffer of {len(buffer)} events...")
        total_processed += len(buffer)

    print(f"\n[STREAMING COMPLETE]")
    print(f"Total events processed: {total_processed}")


# ============================================================================
# ETL PIPELINE
# ============================================================================
# ETL (Extract, Transform, Load) is the foundational technique for data ingestion.
# It ensures data quality through validation and enrichment before loading.
#
# ETL vs ELT:
# - ETL: Transform data before loading (traditional approach)
# - ELT: Load data first, then transform (modern cloud approach)
#
# TOOLS: Apache NiFi, Talend, Informatica, dbt (for ELT)
#
# BEST PRACTICES:
# 1. Data Quality: Validate at ingestion time
# 2. Idempotency: Ensure repeated runs produce same results
# 3. Monitoring: Track data quality metrics
# 4. Error Handling: Log and alert on failures
# 5. Scalability: Design for growth
# ============================================================================

def etl_pipeline_example():
    """
    Complete ETL Pipeline: Extract, Transform, Load
    Shows data quality checks and enrichment
    """
    print("\n" + "=" * 70)
    print("ETL PIPELINE EXAMPLE")
    print("=" * 70)

    # EXTRACT
    print("\n[EXTRACT PHASE]")
    source_data = generate_sample_data(150)

    # ========================================================================
    # SIMULATE REAL-WORLD DATA QUALITY ISSUES
    # ========================================================================
    # In production, data often arrives with problems from upstream systems.
    # This section intentionally corrupts some records to demonstrate how
    # the ETL pipeline handles invalid data through quality checks.
    # ========================================================================

    # ISSUE #1: Introduce missing/null prices (simulates incomplete data)
    # Using .loc with row slice [5:10] to target rows with index 5-10 (6 rows)
    # Sets 'price' column to None for these rows
    # Real-world cause: API timeouts, database corruption, integration bugs
    source_data.loc[5:10, 'price'] = None

    # ISSUE #2: Introduce zero quantities (simulates invalid business data)
    # Using .loc with row slice [15:20] to target rows with index 15-20 (6 rows)
    # Sets 'quantity' column to 0 for these rows
    # Real-world cause: Data entry errors, system glitches, incomplete transactions
    source_data.loc[15:20, 'quantity'] = 0

    # ========================================================================
    # REPORT EXTRACTION RESULTS WITH DATA QUALITY PREVIEW
    # ========================================================================
    print(f"✓ Extracted {len(source_data)} records")

    # Calculate total problematic records before cleaning
    # .isnull().sum() counts how many prices are null/None
    # (source_data['quantity'] == 0).sum() counts how many quantities are zero
    # Adding them gives total records that will fail quality checks
    problem_count = source_data['price'].isnull().sum() + (source_data['quantity'] == 0).sum()
    print(f"  - Records with issues: {problem_count}")
    print(f"    • Missing prices: {source_data['price'].isnull().sum()}")
    print(f"    • Zero quantities: {(source_data['quantity'] == 0).sum()}")

    # TRANSFORM
    print("\n[TRANSFORM PHASE]")

    # ========================================================================
    # 1. DATA QUALITY CHECKS (Lines 183-194)
    # ========================================================================
    # Purpose: Identify and remove records that don't meet business rules
    # This ensures downstream systems only process valid, reliable data
    # ========================================================================
    print("  1. Running data quality checks...")

    # Store the original record count to calculate data quality metrics later
    # This helps measure how much data was rejected vs accepted
    original_count = len(source_data)

    # QUALITY CHECK #1: Remove records with missing/null prices
    # Business Rule: All orders must have a valid price to calculate revenue
    # Method: Filter using pandas .notna() which keeps only non-null values
    # Example: If price is None or NaN, that row is excluded
    source_data = source_data[source_data['price'].notna()]

    # QUALITY CHECK #2: Remove records with zero or negative quantities
    # Business Rule: Orders must have positive quantities to be valid
    # Method: Filter where quantity > 0, excluding zero/negative values
    # Example: quantity=0 means no items ordered, so record is invalid
    source_data = source_data[source_data['quantity'] > 0]

    # Calculate and report how many records failed quality checks
    # This metric is crucial for monitoring data source health
    removed = original_count - len(source_data)
    print(f"     ✓ Removed {removed} invalid records")
    print(f"     ✓ Data quality pass rate: {(len(source_data)/original_count*100):.1f}%")

    # ========================================================================
    # 2. DATA ENRICHMENT (Lines 196-202)
    # ========================================================================
    # Purpose: Add calculated fields and derived attributes that provide
    # additional business value without changing the raw data
    # ========================================================================
    print("  2. Enriching data...")

    # ENRICHMENT #1: Calculate revenue (derived metric)
    # Formula: revenue = quantity × price
    # This pre-calculates the key business metric for faster reporting
    source_data['revenue'] = source_data['quantity'] * source_data['price']

    # ENRICHMENT #2: Extract time-based dimensions from order_date
    # These enable time-series analysis and temporal aggregations
    source_data['year'] = source_data['order_date'].dt.year        # e.g., 2025
    source_data['month'] = source_data['order_date'].dt.month      # e.g., 1-12
    source_data['quarter'] = source_data['order_date'].dt.quarter  # e.g., Q1-Q4

    print("     ✓ Added calculated fields (revenue, year, month, quarter)")

    # ========================================================================
    # 3. DATA STANDARDIZATION (Lines 204-208)
    # ========================================================================
    # Purpose: Normalize formats to ensure consistency across records
    # This prevents issues like "US" vs "us" being treated as different values
    # ========================================================================
    print("  3. Standardizing formats...")

    # STANDARDIZATION #1: Convert country codes to uppercase
    # Ensures consistency: "us", "US", "Us" all become "US"
    source_data['country'] = source_data['country'].str.upper()

    # STANDARDIZATION #2: Apply title case to product names
    # Ensures consistency: "laptop", "LAPTOP", "LaPtOp" all become "Laptop"
    source_data['product'] = source_data['product'].str.title()

    print("     ✓ Standardized text fields (country→UPPER, product→Title)")

    # LOAD
    print("\n[LOAD PHASE]")
    output_file = 'etl_output.csv'
    source_data.to_csv(output_file, index=False)
    print(f"✓ Loaded {len(source_data)} clean records to {output_file}")

    # VALIDATION
    print("\n[VALIDATION & INSIGHTS]")
    print(f"Data Quality Score: {(len(source_data) / original_count * 100):.1f}%")
    print(f"Total Revenue: ${source_data['revenue'].sum():,.2f}")
    print(f"\nTop Products by Revenue:")
    top_products = source_data.groupby('product')['revenue'].sum().sort_values(ascending=False)
    for product, revenue in top_products.items():
        print(f"  {product}: ${revenue:,.2f}")

    return source_data


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" DATA INGESTION PATTERNS DEMONSTRATION ".center(70))
    print("=" * 70)

    # Run batch ingestion
    batch_data = batch_ingestion_example()

    # Run streaming ingestion
    streaming_ingestion_example()

    # Run ETL pipeline
    etl_data = etl_pipeline_example()

    print("\n" + "=" * 70)
    print(" ALL DEMONSTRATIONS COMPLETE ".center(70))
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Batch Ingestion: Cost-effective for scheduled processing")
    print("  • Streaming: Low-latency for real-time use cases")
    print("  • ETL: Ensures data quality through validation and transformation")
    print("\nBest Practices Applied:")
    print("  • Data quality checks (removed invalid records)")
    print("  • Data enrichment (calculated derived fields)")
    print("  • Error handling (validation and logging)")
    print("  • Monitoring (tracked metrics and data quality scores)")
    print("\nOutput files created:")
    print("  • batch_output.csv")
    print("  • etl_output.csv")
    print("\nNext Steps:")
    print("  • Learn Apache Kafka for production streaming")
    print("  • Explore Apache Airflow for workflow orchestration")
    print("  • Study cloud platforms (AWS Glue, Azure Data Factory)")
    print("  • Practice with real-world datasets")