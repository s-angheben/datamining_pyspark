from utils import *

from collaborative_filtering import approx_nearest_neighbors, prepare_model
from src.lib.data_utils import load_utility_matrix, load_user_set
from utils import init_spark, end_session

if __name__ == "__main__":
    # Init spark session
    sc = init_spark("collaborative-filtering")

    utility_matrix = load_utility_matrix(sc)

    model, result_set = prepare_model(utility_matrix, 0.5, 10)
    # print(result_set.head())

    user_set = load_user_set(sc)
    us = user_set.toPandas()
    counter = 0
    for index, row in us.iterrows():
        user_id = row[0]

        # key = result_set.first()
        key = result_set.where(F.col('user_id') == user_id).first()

        sim_users = approx_nearest_neighbors(model, result_set, key, 10)
        print(sim_users.where(F.col('user_id') != user_id).show())

        counter += 1

        if counter == 30:
            break

    # Terminate spark session
    end_session(sc)
