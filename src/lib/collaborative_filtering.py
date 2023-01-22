from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors, VectorUDT
from data_utils import *


def approx_nearest_neighbors(model, rs, key, num=20):
    # Compute the locality sensitive hashes for the input rows, then perform approximate nearest
    # neighbor search.
    # We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    # `model.approxNearestNeighbors(transformedA, key, 2)`
    print("Approximately searching rs for {} nearest neighbors of the user: {}".format(num, key.user_id))
    sim_users = model.approxNearestNeighbors(rs, key.ratings_vector, num)

    print(sim_users.head())

    return sim_users


def add_ratings_vector_format(result_set_df):
    def f(result_set_indexes, rs_size):
        v = Vectors.dense(result_set_indexes)
        return v

    new_f = F.udf(f, VectorUDT())

    result = result_set_df.withColumn("ratings_vector", new_f(F.col("norm_ratings"), F.size(F.col("norm_ratings"))))

    # new_f = F.udf(Vectors.dense, VectorUDT())
    # result = ratings_vector_df.withColumn("ratings_vector", new_f(F.col("ratings_vector")))

    return result


class CollaborativeFiltering(object):

    def __init__(self, sc_):
        self.sc = sc_
        self.queries = {}

    def prepare_model(self):
        utility_matrix = load_utility_matrix(self.sc)
        ut = utility_matrix_create_array(utility_matrix)
        # print(ut.head())

        item_size = ut.count()

        rs = add_ratings_vector_format(ut)
        # print(rs.head())

        q1 = rs.first().ratings_vector
        print(rs.first().user_id, q1)

        brp = BucketedRandomProjectionLSH(inputCol="ratings_vector", outputCol="hashes", bucketLength=2.0,
                                          numHashTables=3)
        model = brp.fit(rs)

        # Feature Transformation
        print("The hashed dataset where hashed values are stored in the column 'hashes':")
        # model.transform(rs).show()

        return model, rs
