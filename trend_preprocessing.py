import sys
# import com.github.mrpowers.spark.daria.sql.functions._

from pyspark.sql.functions import weekofyear
from pyspark.sql import column as col
from pyspark.sql import functions as f
from datetime import datetime
from pyspark.sql import types
from pyspark.sql.types import StructType,StructField, StringType, IntegerType , DoubleType
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types


    

def main(input, output):
    schema = types.StructType([
    types.StructField('date', StringType(), True),
    types.StructField('price', StringType(), True),
    types.StructField('open', StringType(), True),
    types.StructField('high', StringType()),
    types.StructField('low', types.StringType()),
    types.StructField('volume', types.StringType()),
    types.StructField('change', types.StringType()),
])

    df = spark.read.csv(input ,header=False,schema=schema)
    print(df.show())


    df.createOrReplaceTempView('trends')

    trend_weekly=spark.sql("""
                select week, avg(price) price from
                (select date(date) as week, 
                        cast(price as float) as price  
                        from trends 
                )
                group by week order by week desc
              """)


    print(trend_weekly.show(10))
    trend_weekly.write.option("header", "true").csv(output, mode='overwrite')

    

if __name__ == '__main__':
    spark = SparkSession.builder.appName('trend preprocessing').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext

    input = sys.argv[1]
    output = sys.argv[2]

    main(input, output)
