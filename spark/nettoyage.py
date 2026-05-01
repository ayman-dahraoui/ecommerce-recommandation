import os
os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-17"
os.environ["HADOOP_HOME"] = "C:\\hadoop"

from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.functions import col, when, count

conf = SparkConf()
conf.set("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1,com.amazonaws:aws-java-sdk-bundle:1.11.1026")
conf.set("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
conf.set("spark.hadoop.fs.s3a.access.key", "minioadmin")
conf.set("spark.hadoop.fs.s3a.secret.key", "minioadmin")
conf.set("spark.hadoop.fs.s3a.path.style.access", "true")
conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
conf.set("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
conf.set("spark.driver.host", "localhost")
conf.set("spark.driver.memory", "2g")
conf.set("spark.executor.memory", "2g")

# Utiliser ArrayBuffer au lieu de ByteBuffer (moins de RAM)
conf.set("spark.hadoop.fs.s3a.fast.upload", "true")
conf.set("spark.hadoop.fs.s3a.fast.upload.buffer", "array")
conf.set("spark.hadoop.fs.s3a.multipart.size", "5242880")
conf.set("spark.hadoop.fs.s3a.block.size", "5242880")

# Réduire le nombre de partitions
conf.set("spark.sql.shuffle.partitions", "4")

spark = SparkSession.builder \
    .appName("Nettoyage Ecommerce") \
    .config(conf=conf) \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ================================
# 1. LIRE LES DONNÉES
# ================================
print("=== Lecture des données ===")
events = spark.read.csv("s3a://ecommerce-data/events.csv", header=True, inferSchema=True)
print(f"Lignes totales : {events.count()}")

# ================================
# 2. SUPPRIMER LES VALEURS NULLES
# ================================
print("\n=== Suppression des nulls ===")
events_clean = events.dropna(subset=["visitorid", "itemid", "event"])
print(f"Lignes après nettoyage : {events_clean.count()}")

# ================================
# 3. VOIR LES TYPES D'INTERACTIONS
# ================================
print("\n=== Types d'interactions ===")
events_clean.groupBy("event").count().show()

# ================================
# 4. PONDÉRATION DES INTERACTIONS
# ================================
print("\n=== Pondération des interactions ===")
events_weighted = events_clean.withColumn(
    "score",
    when(col("event") == "view", 1)
    .when(col("event") == "addtocart", 2)
    .when(col("event") == "transaction", 3)
    .otherwise(0)
)

events_weighted.show(5)

# ================================
# 5. CONSTRUIRE MATRICE USER x PRODUIT
# ================================
print("\n=== Matrice Utilisateur x Produit ===")
matrice = events_weighted.groupBy("visitorid", "itemid") \
    .agg({"score": "sum"}) \
    .withColumnRenamed("sum(score)", "score_total")

print(f"Nombre de paires utilisateur-produit : {matrice.count()}")
matrice.show(10)

# ================================
# 6. SAUVEGARDER DANS MINIO
# ================================
print("\n=== Sauvegarde dans MinIO ===")
matrice.write.mode("overwrite").parquet("s3a://ecommerce-data/matrice_users_produits")
print("✅ Matrice sauvegardée dans MinIO !")

spark.stop()