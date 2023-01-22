from collaborative_filtering import approx_nearest_neighbors, prepare_model
from data_utils import *
from utils import init_spark, end_session


def get_neighbors(df, model, u):
    user_key = df.where(F.col("user_id") == u.id).first()
    num_neighbors = 30
    neighbors = approx_nearest_neighbors(model, df, user_key, num_neighbors)
    # print(neighbors.show())
    # Retrieve neighbors' user_id filtering out the user_id of the current user
    return neighbors.select('user_id').filter(neighbors.user_id != u.id).collect()


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

    with open(data_path + 'utility_matrix_filled.csv', 'w') as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(utility_matrix.columns)

        # Iterate over all users and all queries in order to fulfill the utility matrix
        count_users = 0
        for u in user_set.rdd.toLocalIterator():

            # Row to be printed in the fulfilled utility matrix

            # Retrieve similar users
            neighbors_ids = get_neighbors(df, model, u)

            # Row to be printed out to the csv file, one per user
            row = [u.id]

            # Iterate over queries
            for q in query_set.rdd.toLocalIterator():

                # Lookup into the dataframe for the rating (if any) given by user  u.id to query q.query_id
                rs_value = utility_matrix.where(F.col("user_id") == u.id).select(F.col(q.query_id))
                value = rs_value.collect()[0][0]

                if value is not None and value != '':
                    row.append(value)
                else:
                    # TODO: recommendation by retrieving neighbors' ratings for the
                    # given query_id and computing the average
                    row.append('')

            writer.writerow(row)
            count_users += 1
            print(count_users)

            if count_users % 10 == 0:
                print('Processed {} users'.format(count_users))

    # print(utility_matrix.head())

    # End spark session
    end_session(sc)


if __name__ == "__main__":
    query_recommendation()
