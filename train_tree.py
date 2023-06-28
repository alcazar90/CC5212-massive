import time
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, lower, regexp_replace
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, NaiveBayes, MultilayerPerceptronClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Create the ArgmentParser
parser = argparse.ArgumentParser(description='Training program')

# We can choose the model
parser.add_argument('--model', type=str, choices=['logistic', 'tree'],
                    default='logistic', help='Tipo de modelo (logistic/tree)')

# Parse the arguments
args = parser.parse_args()

# Start the timer
start_time = time.time()

# Repository with the data
AMAZON = "hdfs://cm:9000/uhadoop2023/manco/proyecto/"
OUTPUT_PATH = "/uhadoop2023/manco/amazon/"

# Create SparkSession object
spark = SparkSession.builder.appName("Manco hello world").getOrCreate()

# Read CSV file from HDFS into a PySpark DataFrame
data = spark.read.csv(AMAZON + "50k_train.csv", header=False, inferSchema=True) \
            .toDF('polarity', 'review_title', 'review_body') \
            .limit(50000)

# Convert columns to string type
data = data.withColumn("review_title", col("review_title").cast("string"))
data = data.withColumn("review_body", col("review_body").cast("string"))

# Combine 'review_title' and 'review_body' columns into 'review_text'
data = data.withColumn('review_text', concat_ws(' ', col("review_title"), col("review_body")))

# Remove punctuation and convert text to lowercase
data = data.withColumn('review_text', lower(regexp_replace('review_text', '[^\sa-zA-Z0-9]', '')))

# Create pipeline for tokenization, count vectorization, IDF, and a classification model
tokenizer = Tokenizer(inputCol='review_text', outputCol='tokens')
countVectorizer = CountVectorizer(inputCol='tokens', outputCol='raw_features')
idf = IDF(inputCol='raw_features', outputCol='transformed_features')
pipeline_otuput = ""

if args.model == 'logistic':
	model = LogisticRegression(labelCol='polarity', featuresCol='transformed_features', maxIter=10, regParam=0.3, elasticNetParam=0.8, family='multinomial')
	pipeline_otuput = "pipeline"
elif args.model == 'tree':
	model = DecisionTreeClassifier(labelCol='polarity', featuresCol='transformed_features')
	pipeline_otuput = "pipeline_tree_30k"
#elif args.model == 'naivebayes':
	#model = GBTClassifier(labelCol='polarity', featuresCol='transformed_features')
	#model = MultilayerPerceptronClassifier(labelCol='polarity', featuresCol='transformed_features', layers=[2, 2, 1], seed=123)
	#model = NaiveBayes(labelCol='polarity', featuresCol='transformed_features')
	#pipeline_otuput = "pipeline_gbt_10k"

pipeline = Pipeline(stages=[tokenizer, countVectorizer, idf, model])

# Fit the pipeline and save to make inference
pipelineModel = pipeline.fit(data)
#pipelineModel.save(OUTPUT_PATH + 'pipeline')
pipelineModel.write().overwrite().save(OUTPUT_PATH + pipeline_otuput)

# Evaluate the model in the training set
predictions = pipelineModel.transform(data)
evaluator = MulticlassClassificationEvaluator(labelCol='polarity', metricName='accuracy')
#evaluator = BinaryClassificationEvaluator(labelCol='polarity', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)

# End the timer
end_time = time.time()
print(f'Elapsed time: {(end_time - start_time):.4f} seconds')

# Print the evaluation metric (e.g. accuracy, auc)
print('accuracy on trainset:', accuracy)
