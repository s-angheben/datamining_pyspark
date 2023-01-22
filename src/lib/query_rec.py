from collaborative_filtering import approx_nearest_neighbors, prepare_model
from data_utils import *
from utils import init_spark, end_session


def get_neighbors(df, model, uid):
    user_key = df.where(F.col("user_id") == uid).first()
    num_neighbors = 30
    neighbors = approx_nearest_neighbors(model, df, user_key, num_neighbors)
    # print(neighbors.show())
    # Retrieve neighbors' user_id filtering out the user_id of the current user
    n_ids = neighbors.select('user_id').filter(neighbors.user_id != uid).collect()
    # return [str(n[0]) for n in n_ids]
    return n_ids


def get_avg_value_neighbors_old(neighbors_ids, qid, ut):
    n_ids = [str(n[0]) for n in neighbors_ids]
    rs_values = ut.filter(ut.user_id.isin(n_ids) & F.col(qid).isNotNull()).select(F.col(qid), F.col('user_id'))

    sum_ratings = 0
    num_ratings = 0
    for rsv in rs_values.collect():
        if rsv[0] is None:
            continue

        print(rsv[0], rsv[1])
        num_ratings += 1
        sum_ratings += int(rsv[0])
    # Compute average value, write '-999' elsewhere just in order to spot missing values
    average_value = str(round(sum_ratings / num_ratings)) if num_ratings > 0 else '-999'
    return average_value


def query_recommendation():
    # Init spark session
    sc = init_spark("query_recommendation")

    # Load utility matrix
    ut = load_utility_matrix(sc)
    ut.createOrReplaceTempView("utility_matrix")

    # Load relational table
    relational_table = load_relational_table(sc)
    relational_table.createOrReplaceTempView("items")

    # Load query set
    query_set = load_query_set(sc)

    # Load user set
    user_set = load_user_set(sc)

    model, df = prepare_model(ut)

    with open(data_path + 'utility_matrix_filled.csv', 'w') as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(ut.columns)

        # Iterate over all users and all queries in order to fulfill the utility matrix
        count_users = 0
        for u in user_set.rdd.toLocalIterator():

            uid = u.id

            # Retrieve similar users
            neighbors_ids = get_neighbors(df, model, uid)
            neighbors_ids_df = sc.createDataFrame(neighbors_ids)

            # Row to be printed out to the csv file, one per user
            row = [uid]

            # Iterate over queries
            for q in query_set.rdd.toLocalIterator():

                qid = q.query_id

                # Lookup into the dataframe for the rating (if any) given by user  u.id to query q.query_id
                rs_value = ut.where(F.col("user_id") == uid).select(F.col(qid))
                value = rs_value.collect()[0][0]

                if value is not None and value != '':
                    row.append(value)
                else:
                    # Recommendation by retrieving neighbors' ratings for query_id, computing the average

                    # Deprecated
                    # average_value = get_avg_value_neighbors(neighbors_ids, qid, ut)

                    try:
                        rsv = ut.join(neighbors_ids_df, 'user_id').select(F.avg(F.col(q.query_id)))
                        average_value = int(rsv.collect()[0][0])
                    except TypeError:
                        # Foo value in order to spot easily missing values
                        average_value = 999

                    row.append(average_value)

            # Print out row to csv file
            writer.writerow(row)

            # Keep track of how many users have been processed
            count_users += 1
            print(count_users)

            if count_users % 10 == 0:
                print('Processed {} users'.format(count_users))

    # print(utility_matrix.head())

    # End spark session
    end_session(sc)


if __name__ == "__main__":
    query_recommendation()
