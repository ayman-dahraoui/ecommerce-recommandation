import os
os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-17"
os.environ["HADOOP_HOME"] = "C:\\hadoop"

from pyspark.sql import SparkSession
from pyspark import SparkConf

conf = SparkConf()
conf.set("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1,com.amazonaws:aws-java-sdk-bundle:1.11.1026")
conf.set("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
conf.set("spark.hadoop.fs.s3a.access.key", "minioadmin")
conf.set("spark.hadoop.fs.s3a.secret.key", "minioadmin")
conf.set("spark.hadoop.fs.s3a.path.style.access", "true")
conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
conf.set("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
conf.set("spark.hadoop.fs.s3a.connection.timeout", "600000")
conf.set("spark.hadoop.fs.s3a.socket.timeout", "600000")
conf.set("spark.hadoop.fs.s3a.attempts.maximum", "3")
conf.set("spark.driver.host", "localhost")

spark = SparkSession.builder \
    .appName("Ecommerce Recommandation") \
    .config(conf=conf) \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("Lecture de events.csv...")
events = spark.read.csv("s3a://ecommerce-data/events.csv", header=True, inferSchema=True)

print(f"Nombre de lignes : {events.count()}")
events.printSchema()
events.show(5)

spark.stop()