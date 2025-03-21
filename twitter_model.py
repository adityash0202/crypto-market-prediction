import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('colour prediction').getOrCreate()
spark.sparkContext.setLogLevel('WARN')
assert spark.version >= '2.3' # make sure we have Spark 2.3+
import numpy as np
import datetime
import pandas as pd

from pyspark.ml import Pipeline
from pyspark.mllib.util import MLUtils
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, SQLTransformer, Normalizer
from pyspark.ml.classification import MultilayerPerceptronClassifier, RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as f


def normalize(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def get_slope_func(x,order=1):
    x=[float(i) for i in x]
    x=normalize(x)
    coeffs = np.polyfit(x, range(len(x)), order)
    slope = coeffs[-2]
    return float(slope)

get_slope = f.udf(get_slope_func, returnType=types.DoubleType())

def get_classes(x):
    if x>0:
        return 1
    else:
        return 0

get_classes = f.udf(get_classes, returnType=types.IntegerType())


def main(input_1, input_2, output):
    trend = spark.read.option("header",True).option("delimiter",",").csv(input_1)
    sentiment = spark.read.option("header",True).option("delimiter",",").csv(input_2)

    print(trend.show())
    print(sentiment.show())
    
    trend.createOrReplaceTempView('trend')
    sentiment.createOrReplaceTempView('sentiment')

    trend=spark.sql("""
                    select week, cast(price as double) price, 
                    LAG(price,1) OVER (ORDER BY week asc) AS price_1,
                    LAG(price,2) OVER (ORDER BY week asc) AS price_2,
                    LAG(price,3) OVER (ORDER BY week asc) AS price_3,
                    LAG(price,4) OVER (ORDER BY week asc) AS price_4,
                    LAG(price,5) OVER (ORDER BY week asc) AS price_5,
                    LAG(price,6) OVER (ORDER BY week asc) AS price_6
                    from trend where date(week)>=date('2021-01-01')
                    """).na.drop()\
                        .withColumn("x",f.array(f.col('price_6'),
                                                f.col('price_5'),
                                                f.col('price_4'),
                                                f.col('price_3'), 
                                                f.col('price_2'), 
                                                f.col('price_1'), 
                                                f.col('price')))

    trend=trend.withColumn("trend", get_slope(f.col("x"))).select('week', 'trend', 'price')
    print(trend.show())
    trend=trend.withColumn('trend_class', get_classes(f.col('trend')))

    print(trend.show())

    sentiment=spark.sql("""
                    select *, 
                    (polarity+polarity_1+polarity_2+polarity_3+polarity_4+polarity_5+polarity_6)/7 as rolling_polarity,
                    (subjectivity+subjectivity_1+subjectivity_2+subjectivity_3+subjectivity_4+subjectivity_5+subjectivity_6)/7 as rolling_subjectivity 
                    from
                    (select date(week) week, 
                    cast(polarity as double) polarity, 
                    cast(subjectivity as double) subjectivity,
                    
                    cast(LAG(polarity,1) OVER (ORDER BY week asc) as double) AS polarity_1,
                    cast(LAG(polarity,2) OVER (ORDER BY week asc) as double) AS polarity_2,
                    cast(LAG(polarity,3) OVER (ORDER BY week asc) as double) AS polarity_3,
                    cast(LAG(polarity,4) OVER (ORDER BY week asc) as double) AS polarity_4,
                    cast(LAG(polarity,5) OVER (ORDER BY week asc) as double) AS polarity_5,
                    cast(LAG(polarity,6) OVER (ORDER BY week asc) as double) AS polarity_6,
                    
                    cast(LAG(subjectivity,1) OVER (ORDER BY week asc) as double) AS subjectivity_1,
                    cast(LAG(subjectivity,2) OVER (ORDER BY week asc) as double) AS subjectivity_2,
                    cast(LAG(subjectivity,3) OVER (ORDER BY week asc) as double) AS subjectivity_3,
                    cast(LAG(subjectivity,4) OVER (ORDER BY week asc) as double) AS subjectivity_4,
                    cast(LAG(subjectivity,5) OVER (ORDER BY week asc) as double) AS subjectivity_5,
                    cast(LAG(subjectivity,6) OVER (ORDER BY week asc) as double) AS subjectivity_6
                    
                    from sentiment where date(week)>=date('2021-01-01'))
                    """).na.drop()\
                        .withColumn("x",f.array(f.col('polarity_6'),
                                                f.col('polarity_5'),
                                                f.col('polarity_4'),
                                                f.col('polarity_3'), 
                                                f.col('polarity_2'), 
                                                f.col('polarity_1'), 
                                                f.col('polarity')))\
                        .withColumn("y",f.array(f.col('subjectivity_6'),
                                                f.col('subjectivity_5'),
                                                f.col('subjectivity_4'),
                                                f.col('subjectivity_3'), 
                                                f.col('subjectivity_2'), 
                                                f.col('subjectivity_1'), 
                                                f.col('subjectivity')))

    sentiment=sentiment.withColumn("slope_polarity", get_slope(f.col("x")))\
                       .withColumn("slope_polarity", get_classes(f.col("slope_polarity")))\
                       .withColumn("slope_subjectivity", get_slope(f.col("y")))\
                       .withColumn("slope_subjectivity", get_classes(f.col("slope_subjectivity")))

    
    join_data=sentiment.join(trend, trend.week==sentiment.week)\
                       .drop(sentiment.week)
                               
    join_data.createOrReplaceTempView('join_data')


    input_cols=['rolling_polarity', 'rolling_subjectivity', 'slope_polarity', 'slope_subjectivity']
    print(input_cols)
    
    label='trend_class'
    assemble_features = VectorAssembler(inputCols = input_cols, outputCol = 'features')

    nor=Normalizer(inputCol='features')
    rf = RandomForestClassifier(featuresCol='features', labelCol=label, numTrees=100, maxDepth=10, seed=42)
    
    pipeline = Pipeline(stages=[assemble_features,nor,rf])

    base = datetime.datetime(2022, 12, 5)
    date_list = [base - datetime.timedelta(days=x) for x in range(30)]

    prediction_df=[]
    for date in date_list:
        train=spark.sql("select * from join_data where week < date'{}'".format(date))
        val=spark.sql("select * from join_data where week between date'{}' and date'{}'".format(date, date+datetime.timedelta(days=7)))

        prediction_model = pipeline.fit(train)
        predictions = prediction_model.transform(val).select('week','trend_class','prediction').toPandas()
        predictions['model_date']=date
        prediction_df.append(predictions)


    prediction_df = pd.concat(prediction_df)
    print(prediction_df.head())
    prediction_df=spark.createDataFrame(prediction_df)
    prediction_df.write.option("header", "true").csv(output, mode='overwrite')


    
    

if __name__ == '__main__':
    spark = SparkSession.builder.appName('twitter model').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext

    trend = sys.argv[1]+'/twitter_trend'
    sentiment = sys.argv[1]+'/twitter_sentiment'
    output = sys.argv[2]

    main(trend, sentiment, output)



