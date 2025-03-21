import subprocess
import time
from pyspark.sql import SparkSession
import sys
error=False

spark = SparkSession.builder.appName('emr orchestration').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

prefix="s3://cmpt732.project/"


print("trend data preprocessing")
subprocess.run(args = ["spark-submit", 
                        prefix+"spark-apps/trend_preprocessing.py", 
                        prefix+"input_data/bitcoin_data.csv", 
                        prefix+"preprocessed_data/twitter_trend"])
print("trend data preprocessing finished")
print("************")

print("tweet data preprocessing")
subprocess.run(args = ["spark-submit", 
                        prefix+"spark-apps/preprocess_tweets.py", 
                        prefix+"input_data/Bitcoin_tweets.csv", 
                        prefix+"input_data/daily_tweets_v2.csv", 
                        prefix+"preprocessed_data/twitter_sentiment"])
print("tweet data preprocessing finished")
print("************")

print("model building")
subprocess.run(args = ["spark-submit", 
                        prefix+"spark-apps/twitter_model.py", 
                        prefix+"preprocessed_data/",
                        prefix+"output_data"])
print("model building finished")
print("************")

print("prediction analysis")
subprocess.run(args = ["spark-submit", 
                        prefix+"spark-apps/prediction_analysis.py", 
                        prefix+"output_data/twitter_prediction.csv/",
                        prefix+"output_data/twitter_prediction_final"])
print("prediction roc finished")
print("************")
