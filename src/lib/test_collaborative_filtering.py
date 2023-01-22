from collaborative_filtering import CollaborativeFiltering, approx_nearest_neighbors
from utils import init_spark, end_session

if __name__ == "__main__":
    # Init spark session
    sc = init_spark("collaborative-filtering")

    cf = CollaborativeFiltering(sc)

    model, result_set = cf.prepare_model()

    sim_users = approx_nearest_neighbors(model, result_set, result_set.first(), 10)
    print(sim_users.head())

    # Terminate spark session
    end_session(sc)
