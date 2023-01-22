from collaborative_filtering import approx_nearest_neighbors, prepare_model
from src.lib.data_utils import load_utility_matrix
from utils import init_spark, end_session

if __name__ == "__main__":
    # Init spark session
    sc = init_spark("collaborative-filtering")

    utility_matrix = load_utility_matrix(sc)

    model, result_set = prepare_model(utility_matrix)

    sim_users = approx_nearest_neighbors(model, result_set, result_set.first(), 10)
    print(sim_users.head())

    # Terminate spark session
    end_session(sc)
