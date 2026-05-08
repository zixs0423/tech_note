---
layout: default
---

- [Database](#database)
  - [SQL](#sql)
    - [Transaction](#transaction)
      - [Concepts](#concepts)
      - [Source](#source)
      - [Code](#code)
    - [CTE (Common Table Expression)](#cte-common-table-expression)
      - [Concepts](#concepts-1)
      - [Source](#source-1)
      - [Code](#code-1)
    - [CTAS (Create Table As Select)](#ctas-create-table-as-select)
      - [Concepts](#concepts-2)
      - [Source](#source-2)
      - [Code](#code-2)
  - [Spark](#spark)
    - [Basic](#basic)
      - [Concepts](#concepts-3)
      - [Source](#source-3)
      - [Code](#code-3)
    - [Operations](#operations)
      - [Concepts](#concepts-4)
      - [Source](#source-4)
      - [Code](#code-4)
    - [Optimization](#optimization)
      - [Concepts](#concepts-5)
      - [Source](#source-5)
      - [Code](#code-5)


# Database

## SQL

### Transaction

#### Concepts

* ACID Properties:

  * Atomicity: Atomicity ensures that a transaction is treated as a single unit of work, and either all operations within the transaction are successfully completed, or none of them are.
  * Consistency: Consistency ensures that a transaction brings the database from one valid state to another. The database must remain in a valid state before and after the transaction.
  * Isolation: Isolation ensures that the operations within a transaction are invisible to other transactions until the transaction is complete.
  * Durability : Durability ensures that once a transaction is committed, its changes are permanent and will survive any system crash or failure.

<br>

#### Source

<br>


#### Code

<br>

---

### CTE (Common Table Expression)

#### Concepts

* A temporary named query that exists only during the execution of that specific command.
* Syntax: Defined using the WITH keyword.

<br>

#### Source

<br>


#### Code

<br>

---


### CTAS (Create Table As Select)

#### Concepts

* Command used to create a new, permanent table based on the results of a SELECT statement. It combines the table creation and data insertion steps into one efficient move.
* Syntax: CREATE TABLE table_name AS SELECT ...

<br>

#### Source

<br>


#### Code

<br>

---

## Spark

### Basic

#### Concepts

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

* Difference between mapreduce and spark:
  * MapReduce is "Disk-First," Spark is "Memory-First."
  * In-Memory Processing: Spark keeps data in RAM between different stages of a job. It doesn't write to the disk unless it absolutely has to (e.g., during a Shuffle or if it runs out of memory).
  * DAG vs. Linear: MapReduce is a linear "Map -> Reduce" chain. Spark creates a DAG (Directed Acyclic Graph). This allows Spark to look at your entire query and optimize the path. 
  * Speed: Because it avoids constant disk I/O, Spark is often 10x to 100x faster than MapReduce for many workloads.

<br>

#### Source

[understanding-spark-dags](https://medium.com/plumbersofdatascience/understanding-spark-dags-b82020503444)
<br>


#### Code

<br>

---

### Operations

#### Concepts

* The result is always a DataFrame when executing spark.sql().
  
* DataFrame: DataFrame = RDD + Schema
  
* RDD (Resilient Distributed Dataset): 
  When you call .rdd on a DataFrame, you are accessing the underlying collection of Row objects distributed across the cluster.
  Unlike a DataFrame, an RDD does not know anything about the names of the columns or the types of data inside it until it’s actually processed. It is "unstructured" compared to the DataFrame API.

* The Schema: the metadata that defines the structure of your data. If the RDD is the "content," the Schema is the "table definition." It contains Column Names, Data Types, and Nullability.
  
* Export data: saveAsTextFile method is one of the original "Actions" in Spark. It takes the data inside an RDD and writes it out to a storage system as plain text files.If you have columns and types, use dataframe.write.csv()

<br>

#### Source

[sql-getting-started](https://spark.apache.org/docs/latest/sql-getting-started.html)

[web-ui](https://spark.apache.org/docs/latest/web-ui.html)

[understanding-spark-ui](https://medium.com/@himanshukotkar007/understanding-spark-ui-b6250d3bdc47)

<br>

#### Code

[spark.py](../code/spark.py)

<br>

---

---

### Optimization

#### Concepts

* Data skew: a condition in distributed computing where data is not distributed **evenly** across the cluster. In a distributed system, your job is only as fast as its slowest task. Power Users/Hot Keys and Null Values may lead to this.
  
* Threshold configuration (spark.sql.autoBroadcastJoinThreshold)
  
  If the statistics for one of the tables show it is smaller than this value, the engine will automatically plan a Broadcast Hash Join.

* Broadcast Hash Join (BHJ):
  
  One of the datasets (the "small" one) is collected at the driver and then sent (broadcast) to every worker node in the cluster.

  Pros: Extremely fast; eliminates the need for shuffling and sorting.

  Cons: Risk of Out of Memory (OOM) errors if the broadcast table is too large.

* Sort Merge Join (SMJ):
  
  Both datasets are re-partitioned (sent to the same node) across the cluster based on the join key.

  Pros: Highly scalable and robust; can handle massive datasets; does not require fitting an entire table into memory.

  Cons: Slower due to the high cost of shuffling data over the network and the CPU cost of sorting.


<br>

#### Source

<br>

#### Code

<br>

---
