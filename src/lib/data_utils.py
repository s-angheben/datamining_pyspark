from utils import *
import re

def load_relational_table(sc):
    relational_table = sc.read.option("header",True)\
                              .option("inferSchema",True) \
                              .csv(data_path + 'relational_table.csv')
    return relational_table


def load_user_set(sc):
    user_set = sc.read.option('lineSep', "\n")\
                      .schema('id STRING')\
                      .text(data_path + 'user_set.csv')
    return user_set


def load_utility_matrix(sc):
    utility_matrix = sc.read.option("header",True)\
                            .csv(data_path + "utility_matrix.csv")
    return utility_matrix


def load_query_set(sc):
    df = sc.read.option('lineSep', "\n")\
                .text(data_path + 'query_set.csv')

    # split the string into an array
    df2 = df.select(F.split(F.col("value"),",").alias("allinfo"))
    # save the first element of the list to a new column (query_id)
    df2 = df2.withColumn('query_id', df2.allinfo[0])
    # save the tail to a new column (query_param)
    df2 = df2.withColumn("query_param", F.expr("slice(allinfo, 2, SIZE(allinfo))"))
    # create the query_string concatenated with AND and save it in query_string
    df2 = df2.withColumn("query_string", F.concat_ws(" AND ",F.col("query_param")))


    df2 = df2.drop("allinfo")

    # print(df2.first().query_string)
    # df2.printSchema()
    # df2.show()
    return df2


def get_query_result_set_ids(sc, query_string, relational_table):
    # add quotes to values between AND
    query_string = re.sub(r'(=)\s*(.*?) AND', r'\1"\2" AND', query_string)
    # add quotes to the last value
    tmp = query_string.split("=")
    last_value = tmp.pop()
    last_value = '"' + last_value + '"'
    tmp.append(last_value)
    query_string = "=".join(tmp)
    print(query_string)

    relational_table.createOrReplaceTempView("items")
    result_set = sc.sql("SELECT * FROM items WHERE " + query_string)

    return result_set



def test():
    sc = init_spark("data_utils")
    utility_matrix = load_utility_matrix(sc)
    user_set = load_user_set(sc)
    relational_table = load_relational_table(sc)
    query_set = load_query_set(sc)

    ## example query
    query1 = query_set.first()
    rs = get_query_result_set_ids(sc, query1, relational_table)
    print(type(rs.collect()))
    print(rs.count())
    rs.show()



if __name__ == "__main__":
    test()
