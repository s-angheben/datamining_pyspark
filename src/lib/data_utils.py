import pandas as pd
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.sql.functions import *
import pyspark.sql.types
from pyspark.sql import Row

def init_spark():
    sc = SparkSession.builder.appName("data_utils")\
                             .config("spark.driver.memory", "16g")\
                             .config("spark.driver.maxResultSize", "4g")\
                             .config("spark.sql.debug.maxToStringFields", "200")\
                             .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
                             .getOrCreate()

    return sc

def load_relational_table(sc):
    relational_table = sc.read.option("header",True)\
                              .option("inferSchema",True) \
                              .csv('../../data/relational_table.csv')
    return relational_table


def load_user_set(sc):
    user_set = sc.read.option('lineSep', "\n")\
                      .schema('id STRING')\
                      .text('../../data/user_set.csv')
    return user_set


def load_utility_matrix(sc):
    utility_matrix = sc.read.option("header",True)\
                            .csv("../../data/utility_matrix.csv")
    return utility_matrix


def load_query_set(sc):
    df = sc.read.option('lineSep', "\n")\
                .text('../../data/query_set.csv')

    # split the string into an array
    df2 = df.select(split(col("value"),",").alias("allinfo"))
    # save the first element of the list to a new column (query_id)
    df2 = df2.withColumn('query_id', df2.allinfo[0])
    # save the tail to a new column (query_param)
    df2 = df2.withColumn("query_param", expr("slice(allinfo, 2, SIZE(allinfo))"))
    # create the query_string concatenated with AND and save it in query_string
    df2 = df2.withColumn("query_string", concat_ws(" AND ",col("query_param")))
    df2 = df2.drop("allinfo")

    # print(df2.first().query_string)
    # df2.printSchema()
    # df2.show()
    return df2


def test():
    sc = init_spark()
    utility_matrix = load_utility_matrix(sc)
    user_set = load_user_set(sc)
    relational_table = load_relational_table(sc)
    query_set = load_query_set(sc)

    query1 = query_set.first()
    relational_table.createGlobalTempView("items")
    sqlDF = sc.sql("SELECT * FROM global_temp.items WHERE " + query1.query_string)
    print(sqlDF.count())
    # sqlDF.show()


if __name__ == "__main__":
    test()
