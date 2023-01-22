from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, MinHashLSH
from pyspark.ml.linalg import Vectors, VectorUDT


from utils import *
from data_utils import *

# https://spark.apache.org/docs/2.2.3/ml-features.html#minhash-for-jaccard-distance

def add_result_set_vector_format(sc, result_set_df, item_size):

    b_size = sc.sparkContext.broadcast(item_size)

    def f(result_set_indexes, rs_size):
        v = Vectors.sparse(b_size.value, result_set_indexes, [1] * rs_size)
        return v

    new_f = F.udf(f, VectorUDT())

    result = result_set_df.withColumn("result_set_vector", new_f(F.col("result_set"), F.size(F.col("result_set"))))

    # new_f = F.udf(Vectors.dense, VectorUDT())
    # result = result_set_df.withColumn("result_set_vector", new_f(F.col("result_set")))

    return result

def add_minhash_to_result_set(sc, rs):
    mh = MinHashLSH(inputCol="result_set_vector", outputCol="hashes", seed=1234, numHashTables=5)
    model = mh.fit(rs)
    rs = model.transform(rs)
    return rs, model



def test():
    sc = init_spark("content-based-filtering")
    utility_matrix = load_utility_matrix(sc)
    relational_table = load_relational_table(sc)
    query_set = load_query_set(sc)
    relational_table.createOrReplaceTempView("items")

    # save_result_set_dataframe(sc, query_set)

    item_size = relational_table.count()

    rs = load_result_set(sc)
    rs = add_result_set_vector_format(sc, rs, item_size)

    q1 = rs.collect()[1].result_set_vector
    print(q1)

    rs_hashed, model = add_minhash_to_result_set(sc, rs)

    db_matches = model.approxSimilarityJoin(rs_hashed, rs_hashed, 0.9)

    print(db_matches.first())
    # result = model.approxNearestNeighbors(rs, q1, 2).collect()
    # print(result)




if __name__ == "__main__":
    test()
