---
layout: default
---

- [Backend](#backend)
- [Java](#java)
  - [JDK, JRE and JVM](#jdk-jre-and-jvm)
  - [Laguage](#laguage)
    - [Basics](#basics)
    - [Methods](#methods)
    - [Class](#class)
  - [Middleware](#middleware)
    - [Kafka](#kafka)
    - [Thrift](#thrift)
    - [Pigeon](#pigeon)


# Backend

# Java

[java](https://www.geeksforgeeks.org/java/java/)

<br>

---

## JDK, JRE and JVM

* Java Development Kit (JDK): a software development kit used to build Java applications. Includes compiler (javac), debugger, and utilities like jar and javadoc.
* Java Runtime Environment (JRE): provides an environment to run Java programs but does not include development tools.
* Java Virtual Machine (JVM): the core execution engine of Java. It is responsible for converting bytecode into machine-specific instructions.

[jkd_jre_jvm](../images/jkd_jre_jvm.png)

[differences-jdk-jre-jvm](https://www.geeksforgeeks.org/java/differences-jdk-jre-jvm/)

* Bytecode: an intermediate, platform-independent code generated when a .java file is compiled into a .class file. This bytecode is executed by the Java Virtual Machine (JVM), enabling Java’s Write Once, Run Anywhere principle. JVM is platform-dependent and bytecode is platform-independent.

[byte-code-in-java](https://www.geeksforgeeks.org/java/byte-code-in-java/)

<br>

---

## Laguage

### Basics

* Output:

  ```
  int num1 = 10, num2 = 20;

  System.out.print("The addition of ");
  System.out.print(num1 + " and " + num2 + " is: ");
  System.out.println(num1 + num2); // The addition of 10 and 20 is: 30

  String[] fruits = {"Apple", "Banana", "Cherry"};

  System.out.println(Arrays.toString(fruits)); // [Apple, Banana, Cherry]
  ```

* For loops: In the context of a standard Java for loop header, 'i++' and '++i' will produce the exact same result.

  ```
  * for (int i = 0; i <= 10; i++) {
              System.out.print(i + " ");
          }
  ```

  But the ouput here will be different:

  ```
  int a = 10;
  System.out.println("Postincrement : " + (a++)); // Postincrement : 10
  System.out.println("Preincrement : " + (++a)); // Preincrement : 12
  ```

  [operators-in-java](https://www.geeksforgeeks.org/java/operators-in-java/)

* Array: 

  ```
  int arr[] = new int[size];
  ```

  [arrays-in-java](https://www.geeksforgeeks.org/java/arrays-in-java/)

<br>

---

### Methods

* Methods:
  
  ```
    public int max(int x, int y) {
      if (x>y) 
        return x;
      else
        return y;
    }
  ```
  Here 'public' is the access modifier, 'int' is the reutrn-type, 'max' is the method-name', 'int x, int y' is the parameter list.

  [methods-in-java](https://www.geeksforgeeks.org/java/methods-in-java/)

* Access Modifiers:
  * Used to control the visibility and accessibility of classes, methods, and variables.
  * Public modifier: It is accessible from anywhere in the program
  * Default modifier: It is accessible only within the same package
  * Protected modifier: It is accessible within the same package and by subclasses
  * Private modifier: It is accessible only within the same class
  * Program: the entire application, every folder within the project.
  * Package: a folder used to group realted classes.
  * File: In Java, a single .java file usually represents one primary class.
    * If you have a class named public class Car, it must be saved in a file named Car.java.
    * However, a single file can actually contain multiple classes, but only one of them can be public.

  [access-modifiers-java](https://www.geeksforgeeks.org/java/access-modifiers-java/)

* Non-access Modifiers:
  * Static: The static keyword means that the entity to which it is applied is available outside any particular instance of the class. That means the static methods or the attributes are a part of the class and not an object.

    ```
    import java.io.*;

    // static variable
    class static_gfg {
        static String s = "GeeksforGeeks"; 
    }
    class GFG {
        public static void main(String[] args)
        {
            // No object required
            System.out.println(
                static_gfg.s); 
        }
    }
    ```

    In this above code sample, we have declared the String as static, part of the static_gfg class. Generally, to access the string, we first need to create the object of the static_gfg class, but as we have declared it as static, we do not need to create an object of static_gfg class to access the string. We can use className.variableName for accessing it.

  * Final: The final keyword indicates that the specific class cannot be extended or a method cannot be overridden.

    [non-access-modifiers-in-java](https://www.geeksforgeeks.org/java/non-access-modifiers-in-java/)

<br>

---

### Class

* Class and Objects: 
  * Class: A class is a blueprint used to create objects that share common properties and behavior.
  * Objects: An object is an instance of a class created to access its data and operations. Each object holds its own state.
  
  ```
  class Student {
      int id;
      String n;

      public Student(int id, String n) {
          this.id = id;
          this.n = n;
      }
  }

  public class Main {
      public static void main(String[] args) {
          Student s1 = new Student(10, "Alice");
          System.out.println(s1.id);
          System.out.println(s1.n);
      }
  }
  ```

  Here the 'Student' is a class and s1 is an object.

* Inhereitance:
  * A subclass can reuse the fields and methods of the parent class without rewriting the code.
  * Use 'extends' expression:

    ```
    // Parent class
    class Animal {
        void sound() {
            System.out.println("Animal makes a sound");
        }
    }

    // Child class
    class Dog extends Animal {
        void sound() {
            System.out.println("Dog barks");
        }
    }

    // Child class
    class Cat extends Animal {
        void sound() {
            System.out.println("Cat meows");
        }
    }

    // Child class
    class Cow extends Animal {
        void sound() {
            System.out.println("Cow moos");
        }
    }

    // Main class
    public class Geeks {
        public static void main(String[] args) {
            Animal a;
            a = new Dog();
            a.sound();  

            a = new Cat();
            a.sound(); 

            a = new Cow();
            a.sound();  
        }
    }
    ```

    [inheritance-in-java](https://www.geeksforgeeks.org/java/inheritance-in-java/)

* Constructors
  * A special member that is called when an object is created.
  * A constructor has the same name as the class.

    ```
    class Student {
        String name;

        // Constructor
        Student(String name) {
            this.name = name;
        }

        void display() {
            System.out.println("Name: " + name);
        }

        public static void main(String[] args) {
            Student s1 = new Student("Vishnu");
            s1.display();
        }
    }
    ```
  
    [constructors-in-java](https://www.geeksforgeeks.org/java/constructors-in-java/)

  * This keyword: this is a keyword that refers to the current object, the object whose method or constructor is being executed. It is mainly used to refer to the current class’s instance variables and methods.

    [java-this-keyword](https://www.geeksforgeeks.org/java/java-this-keyword/)

  * New operator: The new operator instantiates a class by dynamically allocating(i.e, allocation at run time) memory for a new object. The new operator is also followed by a call to a class constructor, which initializes the new object.
  
    [new-operator-java](https://www.geeksforgeeks.org/java/new-operator-java/)

* Abstraction
  * Interface:
    * blueprint that defines a set of methods a class must implement without providing full implementation details.
      * A class must implement all abstract methods of an interface.
      * All variables in an interface are public, static, and final by default.
      * Use 'implements' expression

      ```
      import java.io.*;

      // Interface Declared
      interface testInterface {
        
          // public, static and final
          final int a = 10;

          // public and abstract
          void display();
      }

      // Class implementing interface
      class TestClass implements testInterface {
        
          // Implementing the capabilities of Interface
          public void display(){ 
            System.out.println("Geek"); 
          }
      }

      class Geeks{
          
          public static void main(String[] args){
              
              TestClass t = new TestClass();
              t.display();
              System.out.println(t.a);
          }
      }
      ```

    [interfaces-in-java](https://www.geeksforgeeks.org/java/interfaces-in-java/)
    
  * Abstract Class
    * A class that cannot be instantiated and is designed to be extended by other classes.
    * Difference between interface and abstract class:
      * Interface: Provides full abstraction (though modern Java allows default/static methods). A class can implement multiple interfaces (Multiple Inheritance).
      * Abstract Class: Provides partial abstraction (can have concrete methods). A class can extend only one abstract class (Single Inheritance).

      [abstract-classes-in-java](https://www.geeksforgeeks.org/java/abstract-classes-in-java/)

* Polymorphism
  * Allows objects to behave differently based on their specific class type.
  * Overloading: (Compile-Time Polymorphism/static polymorphism) multiple methods with the same name exist but differ in parameter lists. The method to be called is resolved by the compiler at compile time.
  
    ```
    class Helper {

        // Method with 2 integer parameters
        static int Multiply(int a, int b)
        {
            // Returns product of integer numbers
            return a * b;
        }

        // Method 2
        // With same name but with 2 double parameters
        static double Multiply(double a, double b)
        {
            // Returns product of double numbers
            return a * b;
        }
    }


    // Main class
    class Geeks {
        // Main driver method
        public static void main(String[] args)
        {

            // Calling method by passing
            // input as in arguments
            System.out.println(Helper.Multiply(2, 4));
            System.out.println(Helper.Multiply(5.5, 6.3));
        }
    }
    ```

  * Overriding: (Runtime Polymorphism/dynamic method dispatch) It occurs when a method call is resolved at runtime, and it is achieved when a subclass provides its own implementation of a method already defined in its superclass.

    ```
    // Class 1
    // Helper class
    class Parent {

        // Method of parent class
        void Print() { System.out.println("parent class"); }
    }

    // Class 2
    // Helper class
    class Subclass1 extends Parent {

        // Method
        void Print() { System.out.println("subclass1"); }
    }

    // Class 3
    // Helper class
    class Subclass2 extends Parent {

        // Method
        void Print() { System.out.println("subclass2"); }
    }

    // Class 4
    // Main class
    class Geeks {

        // Main driver method
        public static void main(String[] args)
        {

            // Creating object of class 1
            Parent a;

            // Now we will be calling print methods
            // inside main() method
            a = new Subclass1();
            a.Print();

            a = new Subclass2();
            a.Print();
        }
    }
    ```

    [polymorphism-in-java](https://www.geeksforgeeks.org/java/polymorphism-in-java/)

    * The @Override expression is optional. It is officially called an Annotation. It is considered a "best practice" for two major reasons:
      * Compiler Safety (The Safety Net): If you make a typo (e.g., you write void rol() instead of void role()), the compiler will realize that you intended to override something but failed.
      * Readability: It acts as a clear visual marker for other developers.
    * Overloading is called Compile-Time or static polymorphism because the types of the arguments are written explicitly in the code, the compiler can "link" that specific call to that specific method immediately. Overriding is called Runtime or dynamic Polymorphism because the object can be created in a way the compiler cannot predict. (depends on the input):
    
      ```
      Person p;
      if (new Scanner(System.in).nextInt() > 10) {
          p = new Father();
      } else {
          p = new Son();
      }
      p.role();
      ```

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