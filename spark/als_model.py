import os
os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-17"
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 4g pyspark-shell"

from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer

conf = SparkConf()
conf.set("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1,com.amazonaws:aws-java-sdk-bundle:1.11.1026")
conf.set("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000")
conf.set("spark.hadoop.fs.s3a.access.key", "minioadmin")
conf.set("spark.hadoop.fs.s3a.secret.key", "minioadmin")
conf.set("spark.hadoop.fs.s3a.path.style.access", "true")
conf.set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
conf.set("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
conf.set("spark.driver.host", "localhost")
conf.set("spark.driver.memory", "4g")
conf.set("spark.executor.memory", "4g")
conf.set("spark.hadoop.fs.s3a.fast.upload", "true")
conf.set("spark.hadoop.fs.s3a.fast.upload.buffer", "array")
conf.set("spark.hadoop.fs.s3a.multipart.size", "5242880")
conf.set("spark.hadoop.fs.s3a.block.size", "5242880")
conf.set("spark.sql.shuffle.partitions", "4")
conf.set("spark.driver.maxResultSize", "2g")

spark = SparkSession.builder \
    .appName("ALS Recommandation") \
    .config(conf=conf) \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ================================
# 1. LIRE LA MATRICE + ÉCHANTILLON
# ================================
print("=== Lecture de la matrice depuis MinIO ===")
matrice = spark.read.parquet("s3a://ecommerce-data/matrice_users_produits")

# Utiliser 20% des données
matrice = matrice.sample(fraction=0.2, seed=42)
matrice.cache()
print(f"Nombre de paires (20%) : {matrice.count()}")

# ================================
# 2. ENCODER LES IDs
# ================================
print("\n=== Encodage des IDs ===")
user_indexer = StringIndexer(inputCol="visitorid", outputCol="user_index")
item_indexer = StringIndexer(inputCol="itemid", outputCol="item_index")

matrice = user_indexer.fit(matrice).transform(matrice)
matrice = item_indexer.fit(matrice).transform(matrice)

matrice = matrice.withColumn("user_index", col("user_index").cast("integer"))
matrice = matrice.withColumn("item_index", col("item_index").cast("integer"))
matrice = matrice.withColumn("score_total", col("score_total").cast("float"))

print("Encodage terminé ✅")

# ================================
# 3. SPLIT TRAIN / TEST
# ================================
print("\n=== Split Train/Test (80/20) ===")
train, test = matrice.randomSplit([0.8, 0.2], seed=42)
print(f"Train : {train.count()} lignes")
print(f"Test  : {test.count()} lignes")

# ================================
# 4. ENTRAÎNER LE MODÈLE ALS
# ================================
print("\n=== Entraînement du modèle ALS ===")
print("⏳ Cela peut prendre 10-15 minutes...")

als = ALS(
    maxIter=10,
    regParam=0.1,
    rank=10,
    userCol="user_index",
    itemCol="item_index",
    ratingCol="score_total",
    coldStartStrategy="drop"
)

model = als.fit(train)
print("✅ Modèle ALS entraîné !")

# ================================
# 5. ÉVALUER LE MODÈLE
# ================================
print("\n=== Évaluation du modèle ===")
predictions = model.transform(test)
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="score_total",
    predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)
print(f"RMSE : {rmse:.4f}")

# ================================
# 6. RECOMMANDATIONS
# ================================
print("\n=== Recommandations pour 5 utilisateurs ===")
user_recs = model.recommendForAllUsers(10)
user_recs.show(5, truncate=False)

# ================================
# 7. SAUVEGARDER LE MODÈLE
# ================================
print("\n=== Sauvegarde du modèle dans MinIO ===")
model.write().overwrite().save("data/als_model")
print("✅ Modèle sauvegardé dans MinIO !")

spark.stop()