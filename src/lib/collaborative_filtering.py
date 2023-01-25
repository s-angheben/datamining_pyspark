from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors, VectorUDT
from data_utils import *


def approx_nearest_neighbors(model, rs, key, num=20):
    """
    Given a large dataset and an item, approximately find at most k items which have the closest distance to the item.
    :param model: the trained model
    :param rs: the DataFrame holding the data
    :param key: the item
    :param num: maximum number oof nearest neighbors to be computed
    :return: a DataFrame containing the at most {num} discovered neighbors
    """
    # Compute the locality sensitive hashes for the input rows, then perform approximate nearest
    # neighbor search.
    # We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    # `model.approxNearestNeighbors(transformedA, key, 2)`
    print("Approximately searching rs for {} nearest neighbors of the user: {}".format(num, key.user_id))
    sim_users = model.approxNearestNeighbors(rs, key.ratings_vector, num)

    # print(sim_users.head())

    return sim_users


def add_ratings_vector_format(result_set_df):
    """
    Enrich the DataFrame result set with a column holding a dense Vector with the indexesm to be
    user later in the preparation of the model
    :param result_set_df: the DataFrame containing the result set
    :return: the modified DataFrame
    """
    def f(result_set_indexes, rs_size):
        v = Vectors.dense(result_set_indexes)
        return v

    new_f = F.udf(f, VectorUDT())

    result = result_set_df.withColumn("ratings_vector", new_f(F.col("norm_ratings"), F.size(F.col("norm_ratings"))))

    return result


def prepare_model(utility_matrix, bucket_len=2.0, num_hash_tables=3):
    """
    The bucket length can be used to control the average size of hash buckets (and thus the number of buckets).
    A larger bucket length (i.e., fewer buckets) increases the probability of features being hashed to the same bucket
    (increasing the numbers of true and false positives).
    :param utility_matrix: the DataFrame holding the original utilty matrix
    :param bucket_len: the length of each hash bucket, a larger bucket lowers the false negative rate
    :param num_hash_tables: number of hash tables, where increasing number of hash tables lowers the false negative
    rate, and decreasing it improves the running performance.
    :return: the model and the DataFrame holding the data
    """

    # Enrich the original utility matrix with list of ratings (both normal and normalized)
    ut = utility_matrix_create_array(utility_matrix)
    # print(ut.head())

    # Add a column with a Vector representation of data's indexex
    rs = add_ratings_vector_format(ut)
    # print(rs.head())

    # Bucketed Random Projection is an LSH family for Euclidean distance
    # The input is dense or sparse vectors, each of which represents a point in the Euclidean distance space
    # Hash values in the same dimension are calculated by the same hash function.
    #
    brp = BucketedRandomProjectionLSH(inputCol="ratings_vector", outputCol="hashes", bucketLength=bucket_len,
                                      numHashTables=num_hash_tables)

    # Fits a model to the input dataset
    model = brp.fit(rs)

    # print("The hashed dataset where hashed values are stored in the column 'hashes':")
    # model.transform(rs).show()

    return model, rs
