---
title: "Module 7: Big Data and Cloud Computing"
sidebar_position: 7
---

# Module 7: Big Data and Cloud Computing

**Level:** Intermediate | **Prerequisites:** Module 6 (MLOps)

> When data no longer fits in memory or a single machine, you need distributed computing and cloud infrastructure. This module covers the core concepts, tools, and cloud services used in modern data engineering.

---

## Learning Objectives

- Understand the challenges and architecture of big data systems
- Use Apache Spark for distributed data processing
- Design data pipelines and understand the data lake/warehouse pattern
- Work with cloud storage and compute (AWS, GCP concepts)
- Understand streaming data with Kafka fundamentals
- Apply cloud-native ML services

---

## 7.1 What Is Big Data?

The classic **5 Vs** framework:

| V | Description | Example |
|---|---|---|
| **Volume** | Terabytes/petabytes of data | Clickstream logs |
| **Velocity** | Data generated at high speed | Real-time transactions |
| **Variety** | Structured, semi-structured, unstructured | SQL + JSON + images |
| **Veracity** | Data quality and trustworthiness | Noisy sensor data |
| **Value** | Business insight extracted | Recommendation engine |

**When do you actually need big data tools?**
- Dataset exceeds available RAM
- Processing time is unacceptably slow on one machine
- You need real-time streaming
- Multiple teams need reliable, governed access to shared data

---

## 7.2 Apache Spark

Spark is the industry standard for distributed data processing. It processes data in-memory across a cluster of machines.

### Spark Architecture

```
Driver Program
    └── SparkContext
            ├── Executor (Worker 1)
            ├── Executor (Worker 2)
            └── Executor (Worker N)
```

### PySpark Basics

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when

# Create a Spark session (local mode for development)
spark = SparkSession.builder \
    .appName("MyDataPipeline") \
    .master("local[*]") \
    .getOrCreate()

# Load a CSV into a Spark DataFrame
df = spark.read.csv("data/transactions.csv", header=True, inferSchema=True)

print(f"Row count: {df.count()}")
df.printSchema()
df.show(5)
```

### Transformations and Actions

```python
# Transformations (lazy — not executed until an action is called)
filtered = df.filter(col('amount') > 100)
selected = df.select('user_id', 'amount', 'date')
grouped  = df.groupBy('category').agg(
    avg('amount').alias('avg_amount'),
    count('*').alias('total_transactions')
)

# Actions (trigger execution)
grouped.show()
grouped.write.parquet("output/category_summary/", mode="overwrite")
```

### Spark SQL

```python
# Register as a temp view and use SQL
df.createOrReplaceTempView("transactions")

result = spark.sql("""
    SELECT 
        category,
        COUNT(*) AS num_transactions,
        AVG(amount) AS avg_amount,
        SUM(amount) AS total_revenue
    FROM transactions
    WHERE amount > 0
    GROUP BY category
    ORDER BY total_revenue DESC
""")

result.show()
```

### Spark MLlib

```python
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Prepare features
assembler = VectorAssembler(
    inputCols=['feature1', 'feature2', 'feature3'],
    outputCol='features_raw'
)
scaler = StandardScaler(inputCol='features_raw', outputCol='features')
rf = RandomForestClassifier(numTrees=100, labelCol='label')

pipeline = Pipeline(stages=[assembler, scaler, rf])

train, test = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train)

predictions = model.transform(test)
evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
print(f"Accuracy: {evaluator.evaluate(predictions):.4f}")
```

---

## 7.3 Data Storage Patterns

### Data Lake vs. Data Warehouse

| | Data Lake | Data Warehouse |
|---|---|---|
| **Schema** | Schema-on-read | Schema-on-write |
| **Format** | Raw (JSON, CSV, Parquet) | Structured (SQL tables) |
| **Users** | Data scientists, engineers | Analysts, BI tools |
| **Examples** | AWS S3 + Delta Lake | Snowflake, BigQuery, Redshift |
| **Cost** | Low storage, flexible | Higher, optimized queries |

### File Formats for Big Data

```python
# Parquet — columnar, compressed, fast for analytics
df.write.parquet("s3://my-bucket/data/parquet/", mode="overwrite")

# Reading Parquet (much faster than CSV for large datasets)
df_parquet = spark.read.parquet("s3://my-bucket/data/parquet/")

# Delta Lake — adds ACID transactions to data lakes
# pip install delta-spark
spark = SparkSession.builder \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
    .getOrCreate()

df.write.format("delta").save("s3://my-bucket/delta-table/")
```

---

## 7.4 Cloud Computing Fundamentals

### Major Cloud Providers

| Service Type | AWS | GCP | Azure |
|---|---|---|---|
| Object Storage | S3 | Cloud Storage | Blob Storage |
| Managed Spark | EMR | Dataproc | HDInsight |
| Data Warehouse | Redshift | BigQuery | Synapse |
| ML Platform | SageMaker | Vertex AI | Azure ML |
| Serverless Functions | Lambda | Cloud Functions | Azure Functions |
| Container Orchestration | EKS (Kubernetes) | GKE | AKS |

### Working with AWS S3

```python
import boto3
import pandas as pd
from io import BytesIO, StringIO

s3 = boto3.client('s3')
bucket = 'my-ml-bucket'

# Upload a file
s3.upload_file('local_data.csv', bucket, 'data/local_data.csv')

# Download and read into pandas
response = s3.get_object(Bucket=bucket, Key='data/local_data.csv')
df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))

# List objects in a prefix
objects = s3.list_objects_v2(Bucket=bucket, Prefix='data/')
for obj in objects.get('Contents', []):
    print(obj['Key'], obj['Size'])
```

### Environment Variables for Cloud Credentials

```python
# Always use environment variables — never hardcode credentials
import os

AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
AWS_REGION     = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
```

---

## 7.5 Streaming Data with Kafka

Apache Kafka is a distributed event streaming platform for real-time data pipelines.

### Core Concepts

| Concept | Description |
|---|---|
| **Topic** | Named channel for messages |
| **Producer** | Publishes messages to a topic |
| **Consumer** | Reads messages from a topic |
| **Partition** | Ordered log within a topic |
| **Offset** | Position of a message in a partition |
| **Consumer Group** | Set of consumers sharing load |

### Python Kafka Producer

```python
from kafka import KafkaProducer
import json, time

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
)

# Send sensor readings
for i in range(100):
    event = {
        'sensor_id': 'sensor_01',
        'temperature': 22.5 + i * 0.1,
        'timestamp': time.time(),
    }
    producer.send('sensor-readings', value=event)
    time.sleep(0.1)

producer.flush()
print("Produced 100 events")
```

### Python Kafka Consumer

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'sensor-readings',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
)

for message in consumer:
    data = message.value
    print(f"Received: {data['sensor_id']} temp={data['temperature']:.2f}")
```

### Spark Streaming

```python
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, DoubleType

schema = StructType() \
    .add("sensor_id", StringType()) \
    .add("temperature", DoubleType()) \
    .add("timestamp", DoubleType())

stream_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "sensor-readings") \
    .load() \
    .select(from_json(col("value").cast("string"), schema).alias("data")) \
    .select("data.*")

# Windowed aggregation: average temp per sensor per minute
from pyspark.sql.functions import window
agg = stream_df \
    .groupBy(window("timestamp", "1 minute"), "sensor_id") \
    .avg("temperature")

query = agg.writeStream.outputMode("update").format("console").start()
query.awaitTermination()
```

---

## 7.6 Cloud-Native ML Services

### AWS SageMaker (Conceptual)

```python
# Training a model on SageMaker
import sagemaker
from sagemaker.sklearn import SKLearn

estimator = SKLearn(
    entry_point='train.py',
    role='arn:aws:iam::...',
    instance_type='ml.m5.large',
    framework_version='1.0-1',
)

estimator.fit({'train': 's3://my-bucket/train/'})
predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.t2.medium')
```

---

## Common Mistakes

- Loading the full dataset into memory when only a sample is needed for prototyping
- Not partitioning Spark data — a single large file becomes a bottleneck
- Hardcoding cloud credentials in code
- Using CSV for large datasets (Parquet is 5–10× more efficient)
- Over-engineering small datasets that fit in pandas

---

## Practice Exercises

1. Create a PySpark DataFrame from a CSV, filter rows, and write as Parquet.
2. Use Spark SQL to compute the top 10 customers by total spend from a transactions table.
3. Write a Kafka producer that streams fake sensor data and a consumer that prints received messages.
4. Compare the size and read speed of the same 1M-row dataset as CSV vs. Parquet.
5. Use boto3 to upload a trained model artifact to S3 and download it in a different script.

---

## Further Reading

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [AWS Getting Started](https://aws.amazon.com/getting-started/)
- [Data Engineering Zoomcamp (free)](https://github.com/DataTalksClub/data-engineering-zoomcamp)
