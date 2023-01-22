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

    with open(data_path + 'utility_matrix_filled.csv', 'w') as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(utility_matrix.columns)

        # Iterate over all users and all queries in order to fulfill the utility matrix
        count_users = 0
        for u in user_set.rdd.toLocalIterator():
            row = [u.id]
            for q in query_set.rdd.toLocalIterator():
                # print(u.id, u)
                rs_value = sc.sql("SELECT utility_matrix.{} FROM utility_matrix WHERE utility_matrix.user_id = '{}'".format(q.query_id, u.id))
                value = rs_value.collect()[0][0]
                if value is not None and value != '':
                    row.append(value)
                else:
                    row.append('')
                    # TODO: retrieve key and nearest neighbors
                    # key = df.first()
                    # sim_users = approx_nearest_neighbors(model, df, key, 30)
                    # print(sim_users.head())

            writer.writerow(row)
            count_users += 1

            if count_users % 100 == 0:
                print('Processed {} users'.format(count_users))

    # print(utility_matrix.head())

    # End spark session
    end_session(sc)


if __name__ == "__main__":
    query_recommendation()
