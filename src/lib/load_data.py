import pandas as pd
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.sql.functions import *
import pyspark.sql.types
from pyspark.sql import Row

def init_spark():
    sc = SparkSession.builder.appName("load_data")\
                             .config("spark.driver.memory", "16g")\
                             .config("spark.driver.maxResultSize", "4g")\
                             .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
                             .config("spark.sql.debug.maxToStringFields", "200")\
                             .getOrCreate()

    return sc

def load_relational_table(sc):
    relational_table = sc.read.option("header",True)\
                              .option("inferSchema",True) \
                              .csv('../../data/relational_table.csv')
    # relational_table.show()
    relational_table.printSchema()


    # relational_table.select([count(when(col(c).isNull(), c)).alias(c) for c in relational_table.columns]
    #                         ).show()

    # relational_table.createOrReplaceTempView("items")
    # sqlDF = sc.sql("SELECT * FROM items WHERE DogsAllowed=TRUE")
    # sqlDF.show()

def load_user_set(sc):
    user_set = sc.read.option('lineSep', "\n")\
                      .schema('id STRING')\
                      .text('../../data/user_set.csv')
    # user_set.show()
    user_set.printSchema()


def load_utility_matrix(sc):
    utility_matrix = sc.read.option("header",True)\
                            .csv("../../data/utility_matrix.csv")


def main():
    sc = init_spark()
    load_utility_matrix(sc)
    load_user_set(sc)
    load_relational_table(sc)

if __name__ == "__main__":
    main()
