from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

sqlDF = spark.sql("SELECT * FROM mart.sales")
sqlDF.createOrReplaceTempView("people")

# sqlDF.rdd.saveAsTextFile("people.txt")
sqlDF.write.csv("path/to/output_folder")
