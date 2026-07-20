---
layout: default
---

- [Data](#data)
  - [SQL](#sql)
    - [Language](#language)
      - [CTE (Common Table Expression)](#cte-common-table-expression)
      - [CTAS (Create Table As Select)](#ctas-create-table-as-select)
      - [Drop](#drop)
      - [Window Functions](#window-functions)
    - [RDBMS](#rdbms)
      - [Transaction](#transaction)
      - [MySQL](#mysql)
      - [PostgreSQL](#postgresql)
  - [NoSQL](#nosql)
    - [Key-Value Store](#key-value-store)
      - [Redis](#redis)
  - [Storage](#storage)
    - [Kubernetes](#kubernetes)
    - [Cloud](#cloud)
    - [Hadoop](#hadoop)
      - [HDFS](#hdfs)
      - [YARN](#yarn)
      - [Hive](#hive)
  - [Engines](#engines)
    - [MapReduce](#mapreduce)
    - [Spark](#spark)
      - [Dataframe](#dataframe)
      - [API](#api)
      - [Web UI](#web-ui)
      - [Parameters](#parameters)
      - [Physical Operators](#physical-operators)
      - [Optimization](#optimization)
      - [Engineering praticals](#engineering-praticals)
    - [Presto](#presto)
    - [Doris](#doris)
  - [Pandas](#pandas)
    - [Basic Api](#basic-api)
    - [PyArrow](#pyarrow)
  - [Excel](#excel)
    - [Shortcuts](#shortcuts)
  - [S3](#s3)
  - [Parquet](#parquet)


# Data

## SQL

* Structured Query Language (SQL)

[what-is-sql](https://www.geeksforgeeks.org/sql/what-is-sql/)

<br>

---

### Language
 
#### CTE (Common Table Expression)

* A temporary named query that exists only during the execution of that specific command.
* Syntax: Defined using the WITH keyword.

<br>

---

#### CTAS (Create Table As Select)

* Command used to create a new, permanent table based on the results of a SELECT statement. It combines the table creation and data insertion steps into one efficient move.
* Syntax: CREATE TABLE table_name AS SELECT ...

<br>

---
 
#### Drop

* Drop table: 
  
  ```
  DROP TABLE table_name;
  ```

* Drop partition: 
  
  ```
  Alter TABLE table_name DROP IF EXISTS PARTITION (partition_name=partition);
  ```

[sql-drop-table-statement](https://www.geeksforgeeks.org/sql/sql-drop-table-statement/)

<br>

---

#### Window Functions

* General: 
  
  ```
  SELECT column_name1, 
       window_function(column_name2) 
       OVER (PARTITION BY column_name3 ORDER BY column_name4 ROWS BETWEEN N PRECEDING/CURRENT ROW AND M FOLLOWING) AS new_column
  FROM table_name;
  ```
* Ranking:
  
  Salary | RANK() | DENSE_RANK() | ROW_NUMBER()
  --- | --- | --- | --- 
  50000 | 1 | 1 | 1
  50000 | 1 | 1 | 2
  30000 | 3 | 2 | 3
  20000 | 4 | 3 | 4
  10000 | 5 | 4 | 5

* Difference between the window function and the join operation
  * Use the window function when supported, becuase it's faster. Remember to use 'range between' instead of 'rows between' to avoid collecting the data outside the time window when there is data missing.
  * However, the 'range between' might not be supported to operate on string format on hive sql engine. So use the join operation with a date table as left table to avoid unexpectable data missing problem and be consistent with the ETL expression where most integration is constrained by date.

[window-functions-in-sql](https://www.geeksforgeeks.org/sql/window-functions-in-sql/)

<br>

---

### RDBMS

* Relational Database Management Systems (RDBMS).
* The RDBMS handles the "Front Line". Hadoop handles the "History". When a user buys an item on your website, that transaction goes into an RDBMS (MySQL). It needs to be instant, and it needs to be perfectly accurate (you can't accidentally charge someone twice). Once a day, your company likely runs a job that "exports" all the transactions from the MySQL RDBMS and "dumps" them into Hadoop (HDFS).

[rdbms-full-form](https://www.geeksforgeeks.org/dbms/rdbms-full-form/)

<br>

---

#### Transaction

[sql-transactions](https://www.geeksforgeeks.org/sql/sql-transactions/)

* ACID Properties:
  * Atomicity: Atomicity ensures that a transaction is treated as a single unit of work, and either all operations within the transaction are successfully completed, or none of them are.
  * Consistency: Consistency ensures that a transaction brings the database from one valid state to another. The database must remain in a valid state before and after the transaction.
  * Isolation: Isolation ensures that the operations within a transaction are invisible to other transactions until the transaction is complete.
  * Durability : Durability ensures that once a transaction is committed, its changes are permanent and will survive any system crash or failure.

<br>

---
  
#### MySQL

* OLTP (Online Transaction Processing). Looking up one specific row very fast.
* Spark / Presto /Doris are all OLAP (Online Analytical Processing).
* A classic Relational Database.

<br>

---

#### PostgreSQL

<br>

---

## NoSQL

* Document-Based: JSON-like documents. MongoDB
  * Key-Value Store: Key-Value pairs. Redis
  * Column-Oriented: Columns instead of rows: HBase

  [types-of-nosql-databases](https://www.geeksforgeeks.org/dbms/types-of-nosql-databases/)

<br>

---

### Key-Value Store

#### Redis

<br>

---

## Storage

### Kubernetes

* Kubernetes (k8s)
* While YARN was built specifically to manage Big Data jobs (like Hadoop and Spark), Kubernetes was built to manage Containers (like Docker). Kubernetes is an orchestrator that automates the deployment, scaling, and management of "containerized" applications.
  
[introduction-to-kubernetes-k8s](https://www.geeksforgeeks.org/devops/introduction-to-kubernetes-k8s/)

<br>

---

### Cloud

* Large-scale data centers owned by companies like Amazon (AWS), Google (GCP), or Microsoft (Azure). Instead of your company buying physical servers, bolting them into a rack, and plugging in the power cables, they "rent" virtual versions of those resources over the internet.
* On-Premise: (Private Cloud)
* On-Cloud: (Public Cloud)
  
[on-premises-vs-on-cloud](https://www.geeksforgeeks.org/cloud-computing/on-premises-vs-on-cloud/)

<br>

---

### Hadoop

![](../images/Hadoop_1.png)

![](../images/Hadoop_2.png)

[hadoop-an-introduction](https://www.geeksforgeeks.org/big-data/hadoop-an-introduction/)

* Command:
  * Difference between 'hadoop fs' and 'hdfs dfs':
    * 'hadoop fs': the older, global script. It can interact with the file system, but it is also used to run MapReduce jobs, manage cluster administrative tasks, and interact with other core Hadoop plumbing. 
    * 'hdfs dfs': the modern, dedicated command used specifically to interact with the Hadoop Distributed File System. 
    * The 'fs' stands for File System. The 'dfs' stands for Distributed File System. 'hadoop' and 'hdfs' are the global, all-encompassing commands for the entire ecosystem, they have several "sub-commands" other than 'fs' and 'dfs'. 'hadoop fs' and 'hdfs dfs' are interchangeable in most scenarios.
  * hdfs dfs -get 'hdfs_file_path' 'local_folder_path': To copy files/folders from hdfs store to local file system.
  * hdfs dfs -put 'local_file_path' 'hdfs_folder_path': To copy files/folders from local file system to hdfs store.
  * hdfs dfs -ls 'path': list all the files on hdfs.
  * hdfs dfs -mkdir 'folder_name': create a directory on hdfs.
  
  [hdfs-commands](https://www.geeksforgeeks.org/linux-unix/hdfs-commands/)


<br>

---

#### HDFS

* Hadoop Distributed File System (HDFS)

<br>

---

#### YARN

* Yet Another Resource Negotiator (YARN)
* Acts like the Operating System of Hadoop, managing resources (CPU, RAM) and scheduling jobs so that different engines (not just MapReduce, but also Spark or Hive) can run on the same data at the same time.

<br>

---

#### Hive

* Hive was created (originally by Facebook) to allow people who know SQL to work with data in Hadoop. Without Hive, you’d have to write complex Java code to count rows. It turns SQL into a series of jobs that run on Hadoop. It alse stores Metadata.

<br>

---

## Engines

### MapReduce

* MapReduce breaks every big data problem into exactly two rigid phases:
  * Map: Each node processes a local piece of data and turns it into "key-value pairs" (e.g., counting words in a single document).
  * Reduce: The system shuffles all identical keys to the same node, where they are aggregated (e.g., summing up all instances of the word "apple").
* Word Count Example:
  * The Map Phase (Local Processing)
  
    Input: "Apple Banana Apple"

    Mapper Output: (Apple, 1), (Banana, 1), (Apple, 1)
  
  * The Shuffle & Sort Phase (The Bridge)
  
    Input: Pairs from all over the cluster.

    Output to Reducer A: (Apple, 1), (Apple, 1)

    Output to Reducer B: (Banana, 1)

  * The Reduce Phase (Aggregation)
  
    Reducer A Calculation: 1 + 1 = 2

    Final Output: (Apple, 2), (Banana, 1)

[understanding-spark-dags](https://medium.com/plumbersofdatascience/understanding-spark-dags-b82020503444)

<br>

---

### Spark

* Difference between mapreduce and spark:
  * MapReduce is "Disk-First," Spark is "Memory-First."
  * In-Memory Processing: Spark keeps data in RAM between different stages of a job. It doesn't write to the disk unless it absolutely has to (e.g., during a Shuffle or if it runs out of memory).
  * DAG vs. Linear: MapReduce is a linear "Map -> Reduce" chain. Spark creates a DAG (Directed Acyclic Graph). This allows Spark to look at your entire query and optimize the path. 
  * Speed: Because it avoids constant disk I/O, Spark is often 10x to 100x faster than MapReduce for many workloads.
* HDFS vs. Spark
  * Defination:
    * HDFS (Hadoop Distributed File System): A Storage Engine/Layer. Its only job is to break big files into chunks (blocks) and distribute them across a cluster of hard drives reliably.
      * NameNode (The Manager): Holds the filesystem metadata. It knows where file blocks are stored across the cluster (e.g., "file.parquet Block 1 is on DataNode B, Block 2 is on DataNode C"). It does not store actual file data.
      * DataNode (The Worker): Runs on physical machines and stores the actual raw file blocks on disk. It reads or writes data blocks when requested.
    * Spark: A Compute Engine/Layer. Its only job is to take processing logic (SQL, DataFrames, RDDs), load data into memory/CPU across a cluster, and execute transformations fast.
      * Driver (The Manager): Converts your application code into a execution plan (DAG), asks the cluster resource manager (YARN/Kubernetes/Standalone) for executors, and assigns tasks to them.
      * Executor (The Worker): A JVM process running on a worker node that executes tasks assigned by the Driver, keeps data in RAM/disk cache, and performs computations.
  * Workflow:
    [ Spark Driver ] ── (1. Query File Metadata) ──► [ HDFS NameNode ]
            │                                               │
            │◄── (2. Return Block Locations / Locality) ────┘
            │
            ├── (3. Construct FileScans & RDD Partitions)
            │
            ▼
    [ Task Scheduler ]
            │
            ├── Assigns Task 1 (Local) ──► [ Executor B ] ── (4. Short-Circuit Read) ──► [ DataNode B ]
            │                                                                                │
            └── Assigns Task 2 (Remote) ─► [ Executor C ] ── (4. TCP Block Transfer) ───► [ DataNode D ]
  * Architecture:
    * Co-located Architecture (Traditional On-Premise): Spark Executors and HDFS DataNodes run on the exact same physical servers. Advantage: Enables Data Locality (Process Local). Spark Executors read file blocks directly from the local disk via the local DataNode daemon, completely bypassing network bandwidth overhead.
    * Decoupled Architecture (Modern Cloud Standard): Spark Compute and Storage are 100% physically separated. For instance, running Spark on Kubernetes while reading data from Amazon S3, Google Cloud Storage, or a remote HDFS/Ceph cluster. Advantage: Compute and storage can be scaled independently (e.g., expand storage without paying for idle CPU/RAM).

<br>

---

#### Dataframe
  
* DataFrame: DataFrame = RDD + Schema
* RDD (Resilient Distributed Dataset): 
  
  When you call .rdd on a DataFrame, you are accessing the underlying collection of Row objects distributed across the cluster.
  Unlike a DataFrame, an RDD does not know anything about the names of the columns or the types of data inside it until it’s actually processed. It is "unstructured" compared to the DataFrame API.

* The Schema: the metadata that defines the structure of your data. If the RDD is the "content," the Schema is the "table definition." It contains Column Names, Data Types, and Nullability.

<br>

---

#### API

* Start a Spark Session:
  
  ```python
  spark = SparkSession.builder.appName('app_name').config('config_option', "config_value").getOrCreate()
  ```
  
  Use config to set the configurations like the number of excuters or drivers.

* Run a sql query. The result is a DataFrame.
  
  ```python
  df = spark.sql('sql_query')
  ```

* Register the DataFrame as a SQL temporary view, which can be used in the following sql queries.
  
  ```python
  df.createOrReplaceTempView('view_name')
  ```
  
* Returns the contents of a DataFrame as pandas.DataFrame:
  
  ```python
  df_pandas = df.toPandas()
  ```


* Export data: 
  
  ```python
  df.saveAsTextFile('path')   # takes the data inside an RDD and writes it out to a storage system as plain text files. 
  df.write.csv('path')   # If you have columns and types, use this.
  ```
  
[sql-getting-started](https://spark.apache.org/docs/latest/sql-getting-started.html)
  
[spark.py](../code/spark.py)

<br>

---

#### Web UI

* Jobs: A Job is created every time you call an "Action" in your Spark code (e.g., .collect(), .count(), .saveAsTextFile(), .show())
* Stages: Stages are created based on shuffling boundaries
* Tasks: The number of tasks in a Stage is exactly equal to the number of partitions in the RDD/DataFrame being processed in that Stage. Each Task executes the exact same set of code (the Stage's pipeline) but on a different partition (slice) of the data at the same time.
  * Duration: The total time the task took to complete.
  * GC Time (Garbage Collection Time): The total amount of time the Java Virtual Machine (JVM) spent pausing execution to perform Garbage Collection during that specific task's lifecycle.
  * Input Size/Records: The volume of data (in MB/GB) and the total record count that this task read directly from an external storage system (such as HDFS, S3).
  * Outpu Size/Records: The volume of data and number of records this task wrote out to an external storage system (e.g., saving a DataFrame to Parquet/ORC on S3 or HDFS).
  * Shuffle Read/Size: The amount of serialized intermediate data this task read from other executors (or local disk) at the start of a wide stage.
  * Shuffle Write/Size: The amount of intermediate data this task wrote out to local disk at the end of a stage. (**Inside a single Stage, processing happens in memory (pipelined in RAM). Between Stages, operations transition through local disk via the shuffle mechanism.**)
  * Spill (Memory): The uncompressed, in-memory size of the data that Spark was forced to evict from RAM because the task's execution memory limit was exceeded.
  * Spill (Disk): The actual compressed size of that evicted data once it was written down to the local disk. (Any non-zero value here is a major red flag. Disk spilling severely degrades performance because disk I/O and re-serialization are vastly slower than operating in memory. If you see spilling, you typically need to increase spark.executor.memory, increase the number of partitions, or fix data skew.)

[web-ui](https://spark.apache.org/docs/latest/web-ui.html)

[understanding-spark-ui](https://medium.com/@himanshukotkar007/understanding-spark-ui-b6250d3bdc47)

<br>

---

#### Parameters

* spark.executor.memory (Executor Heap Memory): RAM allocated per executor process. Increase this when tasks throw JVM OOM errors or suffer high GC pause times. (8g – 16g)
* spark.executor.cores: Sets the number of concurrent task slots (CPU cores) assigned to each executor process. (4 to 5 Cores)
* spark.executor.memoryOverhead (Off-Heap Memory): Extra memory allocated outside the JVM heap for overhead (PySpark internal processes, shuffle buffers, netty layers). Crucial if your job crashes with driver/executor container killed by YARN/K8s. (10% of executor RAM or 1g – 2g)
* spark.sql.shuffle.partitions: Controls the number of partitions used when shuffling data for joins and aggregations (Default is 200). If processing a 500GB shuffle dataset with the default 200, each partition is 2.5GB (causing severe spilling). Increase this value (e.g., to 2500) to shrink partition sizes. (Target 100MB – 200MB per partition)
* spark.sql.autoBroadcastJoinThreshold: The maximum size in bytes for a table that Spark will broadcast to all worker nodes during a join, converting a costly Sort-Merge Join (requires shuffle) into a fast Broadcast Hash Join (zero shuffle). Do not set too high, or driver RAM will blow up. (10m (Default) – 50m.)
* spark.sql.adaptive.enabled (Adaptive Query Execution (AQE)): Dynamically re-optimizes query plans at runtime based on real-time stage statistics. (true, Default in Spark 3.x)
* spark.sql.adaptive.skewJoin.enabled: Automatically detects skewed keys in joins during execution and splits them into smaller sub-partitions, preventing single "straggler" tasks from running for hours. (true)

<br>

---

#### Physical Operators

* Exchange (Shuffle): Redistributes data across cluster nodes based on hash keys (Inter-Stage boundary).
* Broadcas: Sends a full copy of a small dataset to all executor nodes.
* Sort: Sorts data within a single partition by key (often using external Timsort).Hash Table BuildHashedRelationConstructs an in-memory look-up table (HashMap) from a dataset.
* Scan: Reads raw files (Parquet, ORC, CSV) from disk/cloud storage into RAM.
* Filter: Drops rows that don't match a predicate condition.
* Project: Selects, renames, or computes specified columns.
* WholeStageCodegen: Collapses multiple pipeline operations into a single compiled Java bytecode loop. Eliminates virtual function call overhead; optimizes CPU cache utilization.
* ColumnarToRow: Converts columnar batch data structures into individual row objects. Bridges vectorized file reads (Parquet/ORC) with row-based execution operators.
* Generate: Expands complex array/map columns into individual rows (explode()). 
* CustomShuffleReader: Dynamically adapts shuffle partition reading (coalescing small partitions or splitting skewed keys (AQE)).
* HashAggregate (Fast & In-Memory): performs aggregations by building an in-memory Hash Map of grouping keys and their running aggregate states. Unbound Memory Requirement and Extremely Fast Speed.
  * As rows stream in, Spark hashes the GROUP BY key.
  * It looks up the key in an in-memory hash map.
  * If the key exists, it updates the running aggregate state (e.g., adds to a running sum). If not, it creates a new entry.Once all rows are processed, it emits the key-value pairs.
* SortAggregate (Slow & Memory-Safe Fallback): requires the input stream to be sorted by the GROUP BY keys first before doing the aggregation. Constant Memory Requirement and Slower Speed.
  * Input data is sorted by the grouping keys (inserting a SortExec node if necessary).
  * Spark streams the sorted rows one-by-one.It keeps a running aggregate state for the current key only.
  * As soon as it encounters a new key, it emits the result for the previous key, clears the state buffer, and starts fresh for the new key.
* SortMergeJoin (SMJ):
  * Workflow: 
    * Shuffle: Both datasets are re-partitioned across the cluster based on the join key. This ensures that records with the same key end up on the same node.
    * Sort: Within each partition, the data is sorted by the join key.
      * Unsorted: For $N$ rows in Dataset A and $M$ rows in Dataset B, a nested loop join takes $O(N \times M)$ comparisons.
      * Sorted: Once both datasets are sorted by the join key, Spark uses a Two-Pointer Merge Algorithm:
    * Merge: The engine iterates through both sorted partitions simultaneously, matching keys (similar to a two-pointer approach).
  * Pros: Highly scalable and robust; can handle massive datasets; does not require fitting an entire table into memory.
  * Cons: Slower due to the high cost of shuffling data over the network and the CPU cost of sorting.
* BroadcastHashJoin (BHJ):
  * Workflow: One of the datasets (the "small" one) is collected at the driver and then sent (broadcast) to every worker node in the cluster. Each worker then builds a **hash table** ([hashing](data_structure_and_algorithms.md#hashing)) of this small dataset in memory and performs the join with its local portion of the large dataset. It uses a hash table for O(1) average-time lookups during the join.
  * Pros: Extremely fast; eliminates the need for shuffling and sorting.
  * Cons: Risk of Out of Memory (OOM) errors if the broadcast table is too large.
  * Threshold configuration (spark.sql.autoBroadcastJoinThreshold): If the statistics for one of the tables show it is smaller than this value, the engine will automatically plan a Broadcast Hash Join.

<br>

---

#### Optimization

* Data skew: a condition in distributed computing where data is not distributed **evenly** across the cluster. In a distributed system, The job is only as fast as its slowest task. Power Users/Hot Keys and Null Values may lead to this.
* Join Strategy: Change SortMergeJoin to BroadcastHashJoin when one table is quite small.
* Change COUNT(DISTINCT col) to APPROX_COUNT_DISTINCT(col):
  * APPROX_COUNT_DISTINCT: a high-performance aggregation function in Spark SQL (and other big data systems) used to estimate the number of unique/distinct values in a column.
    * WorkFlow: 
      * FileScan
      * HashAggregate (Partial HLL Sketch): Every executor task hashes each incoming col value and updates a tiny fixed-size bit array called an HLL Register Sketch (e.g., HyperLogLogPlus buffer).
        * Hashes "user_984321" into a 64-bit binary string:$$\mathbf{0000000101101...0101_2}$$Splits the bit string into two parts:
          * Part 1 (The Register Index): Uses the first $k$ bits to choose a slot in the array (e.g., Slot #3).
          * Part 2 (The Rarity Score): Counts the number of leading zeros in the remaining bits before hitting the first 1. In this example: 00000001... $\rightarrow$ 7 leading zeros.
        * Updates the Register: Goes to Slot #3 and sets its value to $\max(\text{current\_value}, 7)$.
        * Discards the input: The original "user_984321" is immediately thrown away.3. 
        * The Ptinciple: The core intuition relies on coin flipping / probability:If you flip a fair coin, the probability of getting heads on the first flip (1...) is $1/2$ (50%). Getting a tail then heads (01...) is $1/4$ (25%). Getting 7 consecutive tails before a heads (00000001...) is $1/2^8 = 1/256$ (~0.39%).
      * Exchange (Shuffle HLL Registers): Spark only shuffles the tiny HLL sketch buffers (a few kilobytes per partition) across the network.
      * HashAggregate (Merge HLL Registers): Bitwise merges the tiny HLL sketches from all nodes together and applies the HyperLogLog estimation formula to output a single integer.
        * HyperLogLog Estimation Formula on the single merged array $M_{\text{final}}$:$$E = \alpha_m \cdot m^2 \cdot \left( \sum_{i=0}^{m-1} 2^{-M[i]} \right)^{-1}$$
        * $m = 16$ (number of registers).$\alpha_m$ is a bias-correction constant calibrated for $m$ (for $m=16$, $\alpha_m \approx 0.673$).Harmonic Mean: Using $\sum 2^{-M[i]}$ averages the inverted probabilities, preventing rare outlier hashes (like a hash with 30 leading zeros) from inflating the estimate.
      * [ Output Result ]
  * COUNT(DISTINCT col):
    * WorkFlow: FileScan, HashAggregate (Partial 1: group col), Exchange (Shuffle by Hash(col)), HashAggregate (Partial 2: count keys), Exchange (Shuffle by Grouping Key), HashAggregate (Final Result).
    * When you write SELECT b, COUNT(DISTINCT a) FROM table GROUP BY b, Spark's Catalyst Optimizer automatically expands your query into that exact two-stage GROUP BY subquery under the hood: SELECT b, COUNT(1) FROM(SELECT a, b FROM tbale GROUP BY a, b) GROUP BY b
  
<br>

---

#### Engineering praticals

* Backfilling/Re-triggering tasks based on soft dependencies may lead to data inconsistency between previous and current results.
  
  Because when upstream tasks is not ready is not ready in the same/current day but is ready when re-running, the inconsistency/discrepancies may appear.

<br>

---


### Presto

* a "Massively Parallel Processing" (MPP) SQL engine. It was designed by Facebook specifically to make interactive queries lightning fast.
* Memory-to-Memory: Presto was built from the ground up to be "in-memory." It streams data from the disk directly through the network to the next processing stage without ever stopping to write "checkpoints" to the disk.
* No "Startup" Overhead: When you run a Spark job, YARN often has to "request" containers, start up a Spark context, and coordinate executors. Presto is usually a "long-running" service—it’s already awake and waiting for your command, so it starts immediately.
* Optimized for SQL: Spark is a general-purpose engine (it can do ML, Streaming, etc.). Presto is a SQL specialist. It does one thing—SQL—and it does it perfectly.
* The "All or Nothing" Problem: If a Presto query is running on 100 nodes and one node fails or runs out of memory, the entire query fails. Presto does not have "fault tolerance" for individual tasks.
* The "OOM" (Out of Memory) Safety: Because Presto keeps everything in RAM, a "too large" query will crash the Presto worker. Your platform likely has a Rule Engine that says: "If this query looks like it will touch more than X terabytes, send it to Spark so it doesn't crash the Presto cluster."
  
<br>

---

### Doris

* Apache Doris is a modern, real-time data warehouse designed for OLAP (Online Analytical Processing). In the "Data Warehouse" family, if Hive is the old, reliable storage vault, Doris is the high-speed trading floor.
* When you use Spark or Presto, they are Compute-only engines. Doris, however, follows the Storage-Compute Coupled principle. It owns the physical disks and dictates exactly how every byte is written.
* Doris is its own independent system that is designed to replace the need for both RDBMS and Hadoop for analytical work.
* Doris simplifies the "distributed system" concept into two main components:
  * Frontend (FE): The "Brain." It handles user connections, parses your SQL queries, and creates the plan for how to get the data.
  * Backend (BE): The "Muscle." These live on your physical Blades. They store the actual data and do the heavy lifting of calculating sums, averages, and joins.

<br>

---

## Pandas

### Basic Api

[pandas api reference](https://pandas.pydata.org/docs/reference/index.html)

* filter:
  
  ```python
  df[['name', 'age', 'salary']] # filter multiple columns
  df[(df['age'] > 25) & (df['dept'] == 'IT')] # filter with multiple conditions
  df[df['salary'].isna()] # drop the nan rows in certain columns.
  ```

  [pandas.DataFrame.drop](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html#pandas.DataFrame.drop)

* drop:
  
  ```python
  df = df.drop(columns=['B', 'C']) # delete multiple columns
  df = df.drop(index=['B', 'C']) # delete multiple rows
  ```

  [pandas.DataFrame.drop](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html#pandas.DataFrame.drop)

* merge: 
  
  ```python
  df_merged = df_left.merge(df_right, on='key', how='inner') # inner join two dataframes
  ```

  [pandas.DataFrame.merge](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html#pandas.DataFrame.merge)

* groupby:
  
  ```python
  df_grouped = df.groupby(['category', 'region']).sum() # group by mulitple columns and sum the rest columns
  ```

  [pandas.DataFrame.groupby](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html#pandas.DataFrame.groupby)

<br>

---

### PyArrow

* Find the hive table path in hdfs first.
  
   ```sql
  DESCRIBE FORMATTED your_database_name.your_table_name;
  ```
  
* Read the hive table in hdfs directly by PyArrow, without using the hdfs commands like 'get' or any spark sql. (But the spark sql logic must be realized first. Because PyArrow can only pull/get the data but cannot compute or write.)
  
  ```python
  import pyarrow as pa
  import pyarrow.dataset as ds
  import pyarrow.fs as pafs

  # PyArrow automatically picks up host/port from $HADOOP_CONF_DIR/core-site.xml
  hdfs_fs, hive_table_path = pafs.FileSystem.from_uri("/user/hive/warehouse/my_database.db/my_table")

  # Force the dt column to be string format because the pyarrow may automatically recognize it as int format
  partition_schema=pa.schema([('dt', pa.string())])

  # Read directly using the auto-configured filesystem
  dataset = ds.dataset(
      hive_table_path,
      filesystem=hdfs_fs,
      format="orc",
      partitioning=ds.partitioning(
        partition_schema=partition_schema
        flavor="hive"
      )
  )

  df = dataset.to_table(
    filter=ds.feild('dt')>='20260101'
  ).to_pandas()
  ```

  [introduction-to-pyarrow](https://www.geeksforgeeks.org/python/introduction-to-pyarrow/)

<br>

---

## Excel

### Shortcuts

* Hold Control and scroll up or down: Zoom out or in.
  
<br>

---

## S3

* Amazon S3 (Simple Storage Service): an object storage service. Instead of organizing data in a traditional file hierarchy (like your computer's folders) or in structured tables, S3 stores data as "objects" inside containers called "buckets." Each object consists of the file itself, a unique identifier, and metadata.
*  True Decoupling of Compute and Storage: With HDFS, if you run out of storage, you have to spin up more Hadoop nodes, meaning you pay for extra CPU power you might not need. With S3, your data sits safely and cheaply in a managed bucket. You only spin up your query engines (like Hive, Presto, or Spark) when you actually need to compute something, and turn them off when you're done.
  
<br>

---

## Parquet

* Think of a Parquet file (.parquet) as the polar opposite of a CSV. While a CSV stores data row by row, Apache Parquet is an open-source, binary file format that stores data column by column. 
* Read Parquet files:

  ```python
  import pandas as pd

  # Read the entire Parquet file into a DataFrame
  df = pd.read_parquet('data.parquet')

  print(df.head())
  ```
  
<br>

---