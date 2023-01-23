from collaborative_filtering import approx_nearest_neighbors, prepare_model
from content_based_filtering import item_item_prepare_model, item_item_approx_nearest_neighbors
from data_utils import *
from utils import init_spark, end_session
from functools import reduce

def get_neighbors(df, model, uid):
    user_key = df.where(F.col("user_id") == uid).first()
    num_neighbors = 30
    neighbors = approx_nearest_neighbors(model, df, user_key, num_neighbors)
    # print(neighbors.show())
    # Retrieve neighbors' user_id filtering out the user_id of the current user

    # no filter because we want to check the rating of this query by similar users and moreover should be null
    # n_ids = neighbors.select('user_id').filter(neighbors.user_id != uid).collect()
    # return [str(n[0]) for n in n_ids]
    return neighbors


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

def get_similar_items(df, model, query_id):
    query_key = df.where(F.col("query_id") == query_id).first()
    num_similar = 20
    similar_items = item_item_approx_nearest_neighbors(model, df, query_key, num_similar)
    return similar_items

def dropNullColumns(df):
    null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(
        c) for c in df.columns]).collect()[0].asDict()  # 1
    col_to_drop = [k for k, v in null_counts.items() if v > 0]  # 2
    df = df.drop(*col_to_drop)  # 3

    return df

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

    # load already calculated result set
    result_set = load_result_set(sc)

    # user-user model
    model, df = prepare_model(ut)

    # repartition in 4, to parallelize
    df = df.repartition(4)
    print(df.rdd.getNumPartitions())

    # df.printSchema()
    calculated_similar_user = model.approxSimilarityJoin(
        df,
        df,
        50,
        "EuclideanDistance").select(F.col("datasetA.user_id").alias("user_id"),
                                    F.col("datasetB.user_id").alias("user_id_sim"),
                                    F.col("EuclideanDistance")).cache()

    # calculated_similar_user.select(F.max(F.col("EuclideanDistance")), F.min(F.col("EuclideanDistance")), F.avg(F.col("EuclideanDistance"))).show()
    # print(calculated_similar_user.count())
    # print(calculated_similar_user.first())
    # exit(1)

    # item-item model
    item_size = relational_table.count()
    item_item_model, hashed_result_set = item_item_prepare_model(sc, item_size, result_set)


    calculated_similar_item = item_item_model.approxSimilarityJoin(
        hashed_result_set,
        hashed_result_set,
        0.2,
        "JaccardDistance").select(F.col("datasetA.query_id").alias("query_id"),
                                  F.col("datasetB.query_id").alias("query_id_sim"),
                                  F.col("JaccardDistance")).cache()


    # query_user_to_predict = query_set.crossJoin(user_set.select("id"))\
    #                                  .select(F.col("query_id"), F.col("id").alias("user_id"))\
    #                                  .cache()

    print("SIMILAR CALCULATED")

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
    # user_query_to_predict.printSchema()
    # print(user_query_to_predict.first())


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

    print("START PREDICTION")

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

    predicted.printSchema()

    print(predicted.count())
    predicted.write.options(header='True', delimiter=',') \
                   .csv(data_path + "predicted")


    # End spark session
    end_session(sc)


if __name__ == "__main__":
    query_recommendation()
