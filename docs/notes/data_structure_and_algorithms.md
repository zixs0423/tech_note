---
layout: default
---

- [Data Structure and Algorithms](#data-structure-and-algorithms)
  - [Hashing](#hashing)



# Data Structure and Algorithms

[dsa-tutorial-learn-data-structures-and-algorithms](https://www.geeksforgeeks.org/dsa/dsa-tutorial-learn-data-structures-and-algorithms/)

<br>

---

## Hashing

* Hashing refers to the process of generating a small sized output (that can be used as index in a table) from an input of typically large and variable size. Hashing uses mathematical formulas known as hash functions to do the transformation. This technique determines an index or location for the storage of an item in a data structure called Hash Table.
  
  ![hashing](../images/hashing.png)

  [introduction-to-hashing-2](https://www.geeksforgeeks.org/dsa/introduction-to-hashing-2/)

  [hash-table-data-structure](https://www.geeksforgeeks.org/dsa/hash-table-data-structure/)

* Collision resolution techniques:
  * Separate Chaining: make each cell of the hash table point to a linked list of records that have the same hash function value.
  * Linear Probing: The hash table is searched sequentially that starts from the original location of the hash. If in case the location that we get is already occupied, then we check for the next location.
  
  [collision-resolution-techniques](https://www.geeksforgeeks.org/dsa/collision-resolution-techniques/)

* Load factor = Total elements in hash table/ Size of hash table 
  
  [load-factor-and-rehashing](https://www.geeksforgeeks.org/dsa/load-factor-and-rehashing/)

<br>

---