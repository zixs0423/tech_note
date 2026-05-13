---
layout: default
---

- [Backend](#backend)
  - [JavaScript](#javascript)
    - [NPM](#npm)
  - [Middleware](#middleware)
    - [Kafka](#kafka)
    - [Thrift](#thrift)
    - [Pigeon](#pigeon)


# Backend

## JavaScript

### NPM 

* Node Package Manager (NPM)
* The pip for JavaScript

<br>

---

## Middleware

### Kafka

* Apache Kafka is a distributed system used for real-time data streaming.
* Producer: The component that sends or publishes data (events, messages, logs, etc.) to Kafka topics.
* Consumer: The component that reads or subscribes to the data from Kafka topics. It's typically a backend application or microservice that consumes the data for processing, storage, or further actions.
* Broker: These are the Kafka servers that manage the topics and store the data.
* Topic: A Kafka "channel" that holds a specific kind of data (events, logs, messages, etc.).
* Consumer group: A collection of consumers that work together to consume messages from one or more Kafka topics. Kafka ensures that each message from a partition is consumed by only one consumer in the group.

<br>

---

### Thrift

* Apache Thrift is middleware that allows different services (potentially written in different programming languages) to communicate with each other. A framework for cross-language RPC (Remote Procedure Call)

<br>

---

### Pigeon

* Pigeon is typically a message-passing framework or RPC system, often used for building and managing communication between services in a distributed system.

<br>

---