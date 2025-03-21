import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('colour prediction').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.4' # make sure we have Spark 2.4+
from textblob import TextBlob
import subprocess
import os

#os.system("python3 -m pip install textblob")

def preprocessing(lines):
    #words = lines.select(explode(split(lines.value, "t_end")).alias("word"))
    words = lines.na.replace('', None)
    words = words.na.drop()
    words = words.withColumn('word', functions.regexp_replace('text', r'http\S+', ''))
    words = words.withColumn('word', functions.regexp_replace('word', '@\w+', ''))
    words = words.withColumn('word', functions.regexp_replace('word', '#', ''))
    words = words.withColumn('word', functions.regexp_replace('word', 'RT', ''))
    words = words.withColumn('word', functions.regexp_replace('word', ':', ''))
    return words

# text classification
def polarity_detection(text):
    return TextBlob(text).sentiment.polarity

def subjectivity_detection(text):
    return TextBlob(text).sentiment.subjectivity

def text_classification(words):
    # polarity detection
    polarity_detection_udf = functions.udf(polarity_detection, types.StringType())
    words = words.withColumn("polarity", polarity_detection_udf("word"))
    # subjectivity detection
    subjectivity_detection_udf = functions.udf(subjectivity_detection, types.StringType())
    words = words.withColumn("subjectivity", subjectivity_detection_udf("word"))
    return words


def main(input_1, input_2, output):
    data1 = spark.read.option("header",True).option("delimiter",",").csv(input_1)
    data2 = spark.read.option("header",True).option("delimiter",",").csv(input_2)
    
    data1=data1.filter(functions.col('is_retweet')=="False").select('date','text')
    
    words1 = preprocessing(data1)
    words2 = preprocessing(data2)
    
    #text classification to define polarity and subjectivity
    words1 = text_classification(words1)
    words2 = text_classification(words2)

    words1.createOrReplaceTempView('words1')
    words2.createOrReplaceTempView('words2')
    
    spark.sql("""(select date(date) date, polarity, subjectivity, text from words1)
                 union all 
                 (select date(date) date, polarity, subjectivity, text from words2)""").createOrReplaceTempView('words')

    out=spark.sql("""select week, avg(polarity) as polarity, avg(subjectivity) subjectivity from
                 (select date week, polarity, subjectivity from words)
                 group by week having week>=date'2021-01-01'
                 order by week desc""")


    out.write.option("header", "true").csv(output, mode='overwrite')



if __name__ == '__main__':
    spark = SparkSession.builder.appName('pre-process-tweets').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext

    input_1 = sys.argv[1]
    input_2 = sys.argv[2]
    output = sys.argv[3]

    main(input_1, input_2, output)

