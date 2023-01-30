from utils import *

import re
import csv


def load_result_set(sc, fname='result_set.csv'):
    result_set_df = sc.read.option("header", True) \
        .csv(data_path + fname)

    result_set_df = result_set_df.withColumn(
        "result_set",
        F.from_json(F.col("result_set"), "array<integer>")
    )

    return result_set_df


def load_relational_table(sc, fname='relational_table.csv'):
    relational_table = sc.read.option("header", True) \
        .option("inferSchema", True) \
        .csv(data_path + fname)

    relational_table = relational_table.withColumn("index", F.monotonically_increasing_id())
    return relational_table


def load_query_ids_masked(fname='query_ids_masked.csv'):
    query_ids_masked = []
    with open(data_path + fname, 'r') as fin_:
        for r in fin_.read().splitlines():
            query_ids_masked.append(r)

    return query_ids_masked


def load_user_ids_masked(fname='user_ids_masked.csv'):
    user_ids_masked = []
    with open(data_path + fname, 'r') as fin_:
        for r in fin_.read().splitlines():
            user_ids_masked.append(r)

    return user_ids_masked


def load_user_set(sc, fname='user_set.csv'):
    user_set = sc.read.option('lineSep', "\n") \
        .schema('id STRING') \
        .text(data_path + fname)
    return user_set


def load_utility_matrix(sc, fname="utility_matrix.csv"):
    utility_matrix = sc.read.option("header", True) \
        .csv(data_path + fname)
    return utility_matrix


def load_query_user_masked(sc, fname="query_user_masked.csv"):
    utility_matrix = sc.read.option("header", True) \
        .csv(data_path + fname)
    return utility_matrix


def utility_matrix_create_array(df):
    ut = df.select(F.col("user_id"), F.array([c for c in df.columns if c not in {'user_id'}]).alias("ratings"))

    # Function defined by user, to calculate distance between two points on the globe.
    def do_norm(ratings_):
        rated = [int(float(r)) for r in filter(lambda rate: rate is not None, ratings_)]
        # print(rated)
        total = sum(rated)
        # print(total)

        average_rate = total / (1.0 * len(rated))
        # print('Average rate', average_rate)

        norm_ratings = [(int(float(r)) - average_rate) if r is not None else float(0) for r in ratings_]
        # print(norm_ratings)

        return norm_ratings

    # Creating a 'User Defined Function' to normalize ratings
    udf_func = F.udf(do_norm, ArrayType(FloatType()))
    ut = ut.withColumn("norm_ratings", udf_func(ut.ratings))

    # Adding monotonically increasing index
    ut = ut.withColumn("index", F.monotonically_increasing_id())

    # print(ut.first())
    # ut.printSchema()

    return ut


def mask_utility_matrix(ut, num_users=50, num_queries=50):
    """
    Mask the utility matrix by updating the ratings for a range of users and queries,
    determined randomly, in order to predict those values and then perform RMSE by comparing the real ratings
    with the predicted ones
    :param ut: the original utility matrix
    :param num_users: number of users to be included in the masking
    :param num_queries: number of queries to be included in the masking
    :return: the transformed utility matrix, the user and query ids masked
    """
    import random

    start_user = random.randint(0, (ut.count() - num_users - 1))
    end_user = start_user + num_users - 1
    start_query = random.randint(1, len(ut.columns) - num_queries)
    end_query = start_query + num_queries

    # new dataframe derived from previous by adding an index column
    masked_ut = ut.withColumn('index', F.monotonically_increasing_id())

    # List of query ids to be masked
    queries = masked_ut.columns[start_query:end_query]
    # print('Selected {} queries'.format(len(queries)))

    user_ids = [row.user_id for row in
                masked_ut.where(F.col('index').between(start_user, end_user)).select(F.col('user_id')).collect()]

    # print('User IDS', user_ids)
    # print('Selected {} users'.format(len(user_ids)))

    # Slice the utility matrix
    for column_ in queries:
        masked_ut = masked_ut.withColumn(column_, F.when(
            F.col('index').between(start_user, end_user),
            F.lit(None)
        ))

    # sliced_cols = ['user_id'] + queries
    # print('ORIGINAL')
    # print(ut.where(F.col('user_id').isin(user_ids)).select(sliced_cols).show(10))
    # print('MASKED')
    # print(masked_ut.where(F.col('index').between(start_user, end_user)).select(sliced_cols).show(10))
    # print(masked_ut.where(F.col('index').between(start_user, end_user)).select(sliced_cols).show(10))

    # Remove index column
    masked_ut.drop('index')

    return masked_ut, user_ids, queries


def load_query_set(sc, fname='query_set.csv'):
    df = sc.read.option('lineSep', "\n") \
        .text(data_path + fname)

    # split the string into an array
    df2 = df.select(F.split(F.col("value"), ",").alias("allinfo"))
    # save the first element of the list to a new column (query_id)
    df2 = df2.withColumn('query_id', df2.allinfo[0])
    # save the tail to a new column (query_param)
    df2 = df2.withColumn("query_param", F.expr("slice(allinfo, 2, SIZE(allinfo))"))
    # create the query_string concatenated with AND and save it in query_string
    df2 = df2.withColumn("query_string", F.concat_ws(" AND ", F.col("query_param")))

    df2 = df2.drop("allinfo")

    # print(df2.first().query_string)
    # df2.printSchema()
    # df2.show()
    return df2


def get_query_result_set_id(sc, query_string):
    # add quotes to values between AND
    query_string = re.sub(r'(=)\s*(.*?) AND', r'\1"\2" AND', query_string)
    # add quotes to the last value
    tmp = query_string.split("=")
    last_value = tmp.pop()
    last_value = '"' + last_value + '"'
    tmp.append(last_value)
    query_string = "=".join(tmp)

    result_set = sc.sql("SELECT business_id FROM items WHERE " + query_string)

    return result_set


def get_query_result_set_index(sc, query_string):
    # add quotes to values between AND
    query_string = re.sub(r'(=)\s*(.*?) AND', r'\1"\2" AND', query_string)
    # add quotes to the last value
    tmp = query_string.split("=")
    last_value = tmp.pop()
    last_value = '"' + last_value + '"'
    tmp.append(last_value)
    query_string = "=".join(tmp)

    result_set = sc.sql("SELECT index FROM items WHERE " + query_string)

    return result_set


def save_result_set_dataframe(sc, query_set, fname="result_set.csv"):
    # i = query_set.count()
    with open(data_path + fname, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(('query_id', 'result_set'))
        for q in query_set.rdd.toLocalIterator():
            # print(i)
            # i -= 1
            query_result_set = get_query_result_set_index(sc, q.query_string)

            query_result_set_list = query_result_set.select('index').rdd.flatMap(lambda x: x).collect()
            line = (q.query_id, query_result_set_list)
            writer.writerow(line)


def save_random_query_user(df):
    with open(data_path + 'query_user_masked.csv', 'w') as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(df.columns)

        for row in df.rdd.toLocalIterator():
            writer.writerow(row)


def save_masked_utility_matrix(masked_ut=None, uids=None, qids=None):
    """
    Store the masked utility matrix, together with the list of the user ids
    and the query ids in the file system
    :param masked_ut:
    :param uids:
    :param qids:
    """
    with open(data_path + 'utility_matrix_masked.csv', 'w') as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(masked_ut.columns)

        for row in masked_ut.rdd.toLocalIterator():
            writer.writerow(row)

    with open(data_path + 'user_ids_masked.csv', 'w') as f_out:
        f_out.writelines('\n'.join(uids))

    with open(data_path + 'query_ids_masked.csv', 'w') as f_out:
        f_out.writelines('\n'.join(qids))


def test():
    # sc = init_spark("data_utils")
    # utility_matrix = load_utility_matrix(sc)
    # user_set = load_user_set(sc)
    # relational_table = load_relational_table(sc)
    # query_set = load_query_set(sc)
    # relational_table.createOrReplaceTempView("items")

    ## example query
    # query1 = query_set.first()
    # rs = get_query_result_set_index(sc, query1, relational_table)
    # print(type(rs.collect()))
    # print(rs.count())
    # rs.show()

    print(load_user_ids_masked())
    print(load_query_ids_masked())


if __name__ == "__main__":
    test()
