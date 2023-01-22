from collaborative_filtering import approx_nearest_neighbors, prepare_model
from data_utils import *
from utils import init_spark, end_session


def query_recommendation():
    # Init spark session
    sc = init_spark("query_recommendation")

    utility_matrix = load_utility_matrix(sc)
    utility_matrix.createOrReplaceTempView("utility_matrix")
    relational_table = load_relational_table(sc)
    query_set = load_query_set(sc)
    user_set = load_user_set(sc)
    relational_table.createOrReplaceTempView("items")

    model, df = prepare_model(utility_matrix)

    # Iterate over all users and all queries in order to fulfill the utility matrix
    # for u in user_set.rdd.toLocalIterator():
        # for q in query_set.rdd.toLocalIterator():
            # print(u.id, u)
            # user = sc.sql("SELECT {} FROM utility_matrix WHERE utility_matrix.user_id = '{}'".format(q.id, u.id))
            # user = user.withColumn('user_id', F.lit('TEST'))
            # print(user.head())

            # TODO: retrieve key and nearest neighbors
            # key = df.first()
            # sim_users = approx_nearest_neighbors(model, df, key, 30)
            # print(sim_users.head())

    # print(utility_matrix.head())

    # End spark session
    end_session(sc)


if __name__ == "__main__":
    query_recommendation()
