from collaborative_filtering import approx_nearest_neighbors, prepare_model
from content_based_filtering import item_item_prepare_model, item_item_approx_nearest_neighbors
from data_utils import *
from utils import init_spark, end_session
from functools import reduce


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

    # to calculate the result set use
    # save_result_set_dataframe(sc, query_set)

    # load already calculated result set
    result_set = load_result_set(sc)


    # item-item model
    item_size = relational_table.count()
    item_item_model, hashed_result_set = item_item_prepare_model(sc, item_size, result_set)


    calculated_similar_item = item_item_model.approxSimilarityJoin(
        hashed_result_set,
        hashed_result_set,
        0.2,
        "JaccardDistance").select(F.col("datasetA.query_id").alias("query_id"),
                                  F.col("datasetB.query_id").alias("query_id_sim"),
                                  F.col("JaccardDistance"))



    # extract user_query_to_predict
    query_id_list = set(ut.columns) - {"user_id"}
    b_query_id_list = sc.sparkContext.broadcast(query_id_list)

    def fun(x):
        out_put_list = []
        for q in b_query_id_list.value:
            if x[q] == None:
                out_put_list.append({
                    "user_id" : x["user_id"],
                    "query_id": q,
                    # "rating"  : x[q]
                })
        return out_put_list

    user_query_to_predict_rdd = ut.rdd.flatMap(lambda x: fun(x))
    user_query_to_predict = sc.createDataFrame(user_query_to_predict_rdd)


    # extract user_query rated
    def fun2(x):
        out_put_list = []
        for q in b_query_id_list.value:
            if x[q] != None:
                out_put_list.append({
                    "user_id" : x["user_id"],
                    "query_id": q,
                    "rating"  : x[q]
                })
        return out_put_list

    user_query_rated_rdd = ut.rdd.flatMap(lambda x: fun2(x))
    user_query_rated = sc.createDataFrame(user_query_rated_rdd)


    maxloop = 0;
    while(user_query_to_predict.count() != 0 and maxloop != 2):
        print("LOOP: ")
        print(maxloop)
        print()

        # user-user model based on the utility matrix
        user_user_model, hashed_ut = prepare_model(ut)


        calculated_similar_user = user_user_model.approxSimilarityJoin(
            hashed_ut,
            hashed_ut,
            50 + 100 * maxloop,
            "EuclideanDistance").select(F.col("datasetA.user_id").alias("user_id"),
                                        F.col("datasetB.user_id").alias("user_id_sim"),
                                        F.col("EuclideanDistance"))

        # predict the users
        predicted = user_query_to_predict.join(calculated_similar_user, 'user_id')\
                                     .join(calculated_similar_item, 'query_id')\
                                     .cache()

        predicted = predicted.join(user_query_rated,
                                   (predicted["user_id_sim"]  == user_query_rated["user_id"]) &
                                   (predicted["query_id_sim"] == user_query_rated["query_id"])
                                   )\
                             .select(predicted["user_id"], predicted["query_id"], F.col("rating"))\
                             .groupBy(F.col("query_id"), F.col("user_id"))\
                             .agg(F.avg(F.col("rating")).alias("predicted_rating"))


        # merge the predicted rating to the utility matrix
        new_ut = predicted.groupBy(F.col("user_id")).pivot("query_id").avg("predicted_rating")
        new_ut = new_ut.repartition(8)

        ut = ut.alias("ut")
        new_ut = new_ut.alias("new_ut")

        query_id_list = set(new_ut.columns) - {"user_id"}

        ut = ut.join(new_ut, "user_id", how='left')
        ut = ut.repartition(20)
        ut = ut.select([F.col("ut.user_id").alias("user_id")] + [(F.coalesce(F.col("ut."+x), F.col("new_ut."+x))).alias(x) for x in query_id_list])

        # remove the predicted values from the user_query_to_predict dataframe
        user_query_to_predict = user_query_to_predict.subtract(predicted.drop("predicted_rating"))
        # add the predicted values to the user_query_rated dataframe
        user_query_rated = user_query_rated.union(predicted.withColumnRenamed("predicted_rating", "rating"))

        maxloop += 1



    print(user_query_to_predict.count())
    exit(1)
    # # predict the remaining user_query_to_predict with the mean of the corrispettive user rating
    # user_rating_mean = user_query_rated.groupBy(F.col("user_id")).agg(F.avg(F.col("rating")).alias("predicted_rating")).drop("query_id")
    # # predicted = user_query_to_predict.join(user_rating_mean,
    # #                                                   "user_id",
    # #                                                   "left"
    # #                                                   ).select(
    # #                                                       user_query_to_predict["user_id"].alias("user_id"),
    # #                                                       user_query_to_predict["query_id"].alias("query_id"),
    # #                                                       user_rating_mean["predicted_rating"].alias("predicted_rating")
    # #                                                   )

    # ## or user a default value
    # predicted = user_query_to_predict.withColumn("predicted_rating", F.lit("75"))


    # # merge the last predicted rating to the utility matrix
    # query_id_list = set(ut.columns) - {"user_id"}
    # new_ut = predicted.groupBy(F.col("user_id")).pivot("query_id").avg("predicted_rating")
    # new_ut = new_ut.repartition(8)

    # ut = ut.alias("ut")
    # new_ut = new_ut.alias("new_ut")

    # ut = ut.join(new_ut, "user_id", how='left')
    # ut = ut.repartition(20)
    # ut = ut.join(new_ut, "user_id", how='left')

    # ut = ut.select([F.col("ut.user_id").alias("user_id")] + [(F.coalesce(F.col("ut."+x), F.col("new_ut."+x))).alias(x) for x in query_id_list])


    # # write to a single file the fullfilled utility_matrix
    # with open(data_path + 'utility_matrix_filled.csv', 'w') as f_out:
    #     writer = csv.writer(f_out, delimiter=',')
    #     writer.writerow(ut.columns)

    #     for row in ut.rdd.toLocalIterator():
    #         writer.writerow(row)


    # End spark session
    end_session(sc)


if __name__ == "__main__":
    query_recommendation()
