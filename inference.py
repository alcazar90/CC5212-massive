"""
    File: inference.py
    Authors: Cristobal Alcazar, Yerko Garrido, Christopher Stears

    Evaluation steps in the project pipeline to perform predictions.
"""
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, lower, regexp_replace
from pyspark.ml import PipelineModel
#from pyspark.ml.feature import Tokenizer, CountVectorizer, CountVectorizerModel, IDF
from pyspark.ml.classification import LogisticRegressionModel

# Start timer
start_time = time.time()

# Repository with the data
AMAZON = 'hdfs://cm:9000/uhadoop2023/manco/proyecto/'
OUTPUT_PATH = '/uhadoop2023/manco/amazon/'
#FILE = 'mini_test.csv'
#FILE = 'mid_test.csv'
FILE = 'test.csv'

# Create SparkSession object
spark = SparkSession.builder.appName("Manco inference").getOrCreate()

# Load the pipeline...
pipeline_path = OUTPUT_PATH + 'pipeline'
pipeline = PipelineModel.load(pipeline_path)

# Read new CSV file from HDFS into a PySpark DataFrame
new_data = spark.read.csv(AMAZON + FILE, header=False, inferSchema=True) \
                .toDF('polarity', 'review_title', 'review_body')

# Convert columns to string type
new_data = new_data.withColumn('review_title', col('review_title').cast('string'))
new_data = new_data.withColumn('review_body', col('review_body').cast('string'))

# Combine 'review_title' and 'review_body' columns into 'review_text'
new_data = new_data.withColumn('review_text', concat_ws(' ', col("review_title"), col("review_body")))

# Remove punctuation and convert text to lowercase using the same tokenizer preprocessing
new_data = new_data.withColumn('review_text', lower(regexp_replace('review_text', '[^\sa-zA-Z0-9]', '')))

# Use the pipeline to get the predictions
predictions = pipeline.transform(new_data)

# Save the output in the folder OUTPUT 
#predictions.select('polarity', 'prediction', 'review_text').write.csv(OUTPUT_PATH + 'predictions')
predictions.select('polarity', 'prediction', 'review_text').write.mode('overwrite').csv(OUTPUT_PATH + 'predictions')

# End the timer
end_time = time.time()
print(f'Elapsed time: {(end_time - start_time):.4f} seconds')
print('Inference completed. Predictions saved to', OUTPUT_PATH)

