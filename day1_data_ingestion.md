# Data Ingestion Patterns Demo

A beginner-friendly guide to understanding data ingestion patterns with hands-on examples. This project demonstrates three core data engineering concepts: Batch Ingestion, Streaming Ingestion, and ETL Pipelines.

## Table of Contents
- [What is Data Ingestion?](#what-is-data-ingestion)
- [Architecture Overview](#architecture-overview)
- [Ingestion Patterns](#ingestion-patterns)
- [Getting Started](#getting-started)
- [Understanding the Code](#understanding-the-code)
- [Key Concepts](#key-concepts)

## What is Data Ingestion?

Data ingestion is the process of collecting, importing, and processing data from various sources into a storage system where it can be accessed and analyzed. Think of it as the "front door" of your data infrastructure.

## Architecture Overview

```mermaid
graph TB
    subgraph Sources["Data Sources"]
        DB[(Databases)]
        API[APIs]
        Files[Files/Logs]
        Sensors[IoT Sensors]
    end

    subgraph Ingestion["Ingestion Layer"]
        Batch[Batch Ingestion<br/>Scheduled Processing]
        Stream[Streaming Ingestion<br/>Real-time Processing]
    end

    subgraph Processing["ETL Processing"]
        Extract[Extract<br/>Pull Data]
        Transform[Transform<br/>Clean & Enrich]
        Load[Load<br/>Store Data]
    end

    subgraph Storage["Storage Layer"]
        DW[(Data Warehouse)]
        Lake[(Data Lake)]
        CSV[CSV Files]
    end

    DB --> Batch
    API --> Batch
    Files --> Batch
    Sensors --> Stream
    API --> Stream

    Batch --> Extract
    Stream --> Extract
    Extract --> Transform
    Transform --> Load
    Load --> DW
    Load --> Lake
    Load --> CSV

    style Batch fill:#e1f5ff
    style Stream fill:#fff4e1
    style Transform fill:#e8f5e9
```

## Ingestion Patterns

### 1. Batch Ingestion

**What:** Process large volumes of data at scheduled intervals (hourly, daily, weekly)

**When to use:**
- Historical analysis and reporting
- Large datasets that don't need immediate processing
- Scheduled/periodic updates (daily reports, monthly aggregations)
- Complex transformations

```mermaid
sequenceDiagram
    participant Source as Data Source
    participant Scheduler as Scheduler
    participant Batch as Batch Processor
    participant Storage as Storage

    Note over Scheduler: Every 24 hours
    Scheduler->>Batch: Trigger batch job
    Batch->>Source: Extract all records
    Source-->>Batch: 100,000 records
    Note over Batch: Transform all records
    Batch->>Storage: Load all records
    Note over Storage: Data available
```

**Pros:** Efficient for large volumes, simpler to implement, lower cost
**Cons:** Not real-time, delays between data creation and availability

### 2. Streaming Ingestion

**What:** Process data continuously in real-time as events arrive

**When to use:**
- Real-time analytics and dashboards
- Time-sensitive applications (fraud detection, alerts)
- Continuous data sources (IoT sensors, user activity)
- When data value decreases rapidly over time

```mermaid
sequenceDiagram
    participant Source as Event Source
    participant Stream as Stream Processor
    participant Buffer as Micro-batch Buffer
    participant Storage as Storage

    loop Continuous
        Source->>Stream: Event arrives
        Stream->>Buffer: Add to buffer
        alt Buffer full (every 10 events)
            Buffer->>Stream: Process micro-batch
            Stream->>Storage: Load batch
            Note over Storage: Data available<br/>within seconds
        end
    end
```

**Pros:** Low latency, immediate insights, continuous processing
**Cons:** More complex infrastructure, higher cost, harder to debug

### 3. ETL Pipeline

**What:** Extract, Transform, Load - a structured approach to data movement with quality checks

```mermaid
flowchart LR
    subgraph Extract
        Source[(Source<br/>Database)]
    end

    subgraph Transform
        QC[Data Quality<br/>Checks]
        Enrich[Data<br/>Enrichment]
        Standard[Data<br/>Standardization]
    end

    subgraph Load
        Target[(Target<br/>Storage)]
    end

    Source --> QC
    QC -->|Valid Data| Enrich
    QC -.->|Invalid Data| Reject[Rejected Records]
    Enrich --> Standard
    Standard --> Target

    style QC fill:#ffebee
    style Enrich fill:#e8f5e9
    style Standard fill:#e3f2fd
    style Reject fill:#f5f5f5
```

**Key Steps:**
1. **Extract:** Pull data from source
2. **Transform:**
   - Data Quality Checks (remove nulls, validate business rules)
   - Data Enrichment (add calculated fields like revenue)
   - Data Standardization (normalize formats)
3. **Load:** Write to destination storage

## Getting Started

### Prerequisites
```bash
pip install pandas
```

### Running the Demo
```bash
python day1_data_ingestion.py
```

### Expected Output
The script will generate:
- `batch_output.csv` - Results from batch processing
- `etl_output.csv` - Results from ETL pipeline
- Console output showing each step of the process

## Understanding the Code

### Data Flow Example

```mermaid
flowchart TD
    Start([Generate Sample Data]) --> Extract

    subgraph Batch["Batch Processing"]
        Extract[Extract 100 Records]
        Transform[Transform:<br/>Calculate Revenue<br/>Add Metadata<br/>Filter Invalid]
        BatchLoad[Load to CSV]
    end

    subgraph Stream["Streaming Processing"]
        Event[Event Arrives]
        MicroBatch[Buffer 10 Events]
        StreamProcess[Process Buffer]
        Metrics[Calculate Metrics]
    end

    subgraph ETL["ETL Pipeline"]
        ETLExtract[Extract 150 Records]
        Quality[Quality Checks:<br/>Remove Nulls<br/>Validate Quantity]
        Enrich[Enrich:<br/>Add Revenue<br/>Add Date Fields]
        ETLLoad[Load Clean Data]
    end

    Extract --> Transform --> BatchLoad
    Event --> MicroBatch --> StreamProcess --> Metrics
    ETLExtract --> Quality --> Enrich --> ETLLoad

    style Batch fill:#e1f5ff
    style Stream fill:#fff4e1
    style ETL fill:#e8f5e9
```

### Key Code Sections

**Batch Ingestion (Lines 55-128):**
- Processes all 100 records at once
- Vectorized operations for efficiency
- Aggregations across entire dataset

**Streaming Ingestion (Lines 153-258):**
- Processes events one-by-one
- Uses micro-batching (buffer of 10)
- Simulates real-time with 100ms delays

**ETL Pipeline (Lines 264-398):**
- Introduces data quality issues
- Demonstrates quality checks
- Shows data enrichment and standardization

## Key Concepts

### Micro-batching
```mermaid
graph LR
    E1[Event 1] --> Buffer
    E2[Event 2] --> Buffer
    E3[Event 3] --> Buffer
    E10[Event 10] --> Buffer
    Buffer --> Process[Process 10 events together]
    Process --> Clear[Clear Buffer]
    Clear --> Buffer

    style Buffer fill:#fff4e1
    style Process fill:#e8f5e9
```

Small groups of events processed together for efficiency, balancing latency and throughput.

### Data Quality Checks
```mermaid
flowchart LR
    Input[Raw Data<br/>150 records] --> Check1{Price<br/>not null?}
    Check1 -->|Yes| Check2{Quantity<br/> > 0?}
    Check1 -->|No| Reject1[Reject]
    Check2 -->|Yes| Valid[Valid Data<br/>138 records]
    Check2 -->|No| Reject2[Reject]

    style Valid fill:#e8f5e9
    style Reject1 fill:#ffebee
    style Reject2 fill:#ffebee
```

### When to Use Each Pattern

| Pattern | Latency | Complexity | Cost | Use Case |
|---------|---------|------------|------|----------|
| **Batch** | Hours/Days | Low | Low | Daily reports, historical analysis |
| **Streaming** | Seconds | High | High | Fraud detection, live dashboards |
| **ETL** | Varies | Medium | Medium | Data warehousing, quality-critical data |


## Real-World Examples

- **Batch:** Bank processes end-of-day transactions, Netflix generates viewing reports
- **Streaming:** Uber tracks driver locations, Twitter updates trending topics
- **ETL:** Healthcare systems load patient records, E-commerce sites sync inventory

## Further Reading

- [Batch vs Stream Processing](https://aws.amazon.com/streaming-data/)
- [ETL Best Practices](https://cloud.google.com/architecture/etl-best-practices)
- [Lambda Architecture](https://en.wikipedia.org/wiki/Lambda_architecture)

## License

This is an educational demo project for learning data ingestion concepts.
