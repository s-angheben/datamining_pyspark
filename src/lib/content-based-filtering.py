from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, MinHashLSH

from utils import *
from data_utils import *

# https://spark.apache.org/docs/2.2.3/ml-features.html#minhash-for-jaccard-distance



def create_result_set_dataframe(sc, query_set, relational_table):
    query_ids   = []
    result_sets = []
    for row in query_set.rdd.toLocalIterator():
        print(row)
        print(row.query_string)
        query_ids.append(row.query_id)

        result_set = get_query_result_set_ids(sc, row.query_string, relational_table)
        result_sets.append(result_set.collect())

    result_set_df = sc.createDataFrame(zip(query_ids, result_sets),
                                       ['query_id', 'result_set'])
    result_set_df.show()
    result_set_df.printSchema()


def test():
    sc = init_spark("content-based-filtering")
    utility_matrix = load_utility_matrix(sc)
    relational_table = load_relational_table(sc)
    query_set = load_query_set(sc)

    create_result_set_dataframe(sc, query_set, relational_table)


if __name__ == "__main__":
    test()
