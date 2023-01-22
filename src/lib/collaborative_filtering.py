from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col

from utils import *
from data_utils import *


def add_ratings_vector_format(sc, result_set_df, item_size):
    b_size = sc.sparkContext.broadcast(item_size)

    def f(result_set_indexes, rs_size):
        # TODO: sparse --> dense
        # v = Vectors.dense(result_set_indexes[1:])
        v = Vectors.sparse(b_size.value, result_set_indexes, [1] * rs_size)
        return v

    new_f = F.udf(f, VectorUDT())

    result = result_set_df.withColumn("ratings_vector", new_f(F.col("norm_ratings"), F.size(F.col("norm_ratings"))))

    # new_f = F.udf(Vectors.dense, VectorUDT())
    # result = ratings_vector_df.withColumn("ratings_vector", new_f(F.col("ratings_vector")))

    return result


def do_collaborative_filtering():
    sc = init_spark("collaborative-filtering")
    utility_matrix = load_utility_matrix(sc)
    ut = utility_matrix_create_array(utility_matrix)

    print(ut.head())

    item_size = ut.count()

    rs = add_ratings_vector_format(sc, ut, item_size)

    q1 = rs.first().ratings_vector
    # print(q1)

    # TODO: add ratings column to the dataframe
    brp = BucketedRandomProjectionLSH(inputCol="ratings_vector", outputCol="hashes", bucketLength=2.0,
                                      numHashTables=3)
    model = brp.fit(ut)

    # Feature Transformation
    print("The hashed dataset where hashed values are stored in the column 'hashes':")
    model.transform(ut).show()

    # Compute the locality sensitive hashes for the input rows, then perform approximate nearest
    # neighbor search.
    # We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    # `model.approxNearestNeighbors(transformedA, key, 2)`
    print("Approximately searching rs for 2 nearest neighbors of the key:")
    model.approxNearestNeighbors(rs, q1, 2).show()

    end_session(sc)


if __name__ == "__main__":
    do_collaborative_filtering()
