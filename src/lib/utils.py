import pandas as pd
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql import Row

data_path = "../../data/"


def init_spark(name):
    sc = SparkSession.builder.appName(name) \
        .config("spark.driver.memory", "16g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.sql.debug.maxToStringFields", "200") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    return sc


def end_session(sc):
    sc.stop()
