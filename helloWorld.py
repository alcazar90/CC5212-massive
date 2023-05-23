from pyspark.sql import SparkSession
import pandas as pd

# 
AMAZON = "hdfs://cm:9000/uhadoop2023/manco/proyecto/"

# create SparkSession object
spark = SparkSession.builder.appName("Manco hello world").getOrCreate()

# read CSV file from HDFS into a PySpark DataFrame
data = spark.read.csv(AMAZON + "test.csv",
        header=True, inferSchema=True)

# convert PySpark DataFrame to pandas Dtaf
data_pandas = data.toPandas()

#data_pandas = data_pandas.dropna()

print(f"Ultimas 10 filas test.csv\n {data_pandas.tail(n=10)}")





