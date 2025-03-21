from pyspark.sql import SparkSession, functions, types
import sys
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def main(input, output):
    df = spark.read.option("header",True).option("delimiter",",").csv(input)

    df.createOrReplaceTempView('predictions_df')
    predictions_df=spark.sql("""select week, cast(trend_class as double) trend_class, cast(prediction as double) as prediction, 
                             model_date, cast(datediff(day,date(model_date),date(week)) as integer) as lag
                             from predictions_df
                         """)

    print(predictions_df.show())
    
    roc_evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='trend_class', metricName='areaUnderROC')
    roc_ls=[]
    for r in range(8):
        roc = roc_evaluator.evaluate(predictions_df.filter(predictions_df['lag']==float(r)))
        roc_ls.append(roc)
    
    print(roc_ls)
    predictions_df.write.option("header", "true").csv(output, mode='overwrite')


if __name__ == '__main__':
    spark = SparkSession.builder.appName('prediction analysis').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext

    input = sys.argv[1]
    output = sys.argv[1]
    
    main(input, output)
    #prediction analysis
