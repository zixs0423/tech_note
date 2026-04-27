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