from evaluate import *
from utils import init_spark, end_session
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


def als():
    print("Running als algorithm")
    sc = init_spark("query_recommendation")
    utility_matrix = load_utility_matrix(sc)


    ut = utility_matrix.repartition(8)

    query_set = load_query_set(sc)
    query_set = query_set.withColumn("query_id_index", F.monotonically_increasing_id())

    user_set = load_user_set(sc)
    user_set = user_set.select(F.col("id").alias("user_id")).withColumn("user_id_index", F.monotonically_increasing_id())


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

    user_query_to_predict = user_query_to_predict.join(user_set,  'user_id')\
                                                 .join(query_set, 'query_id')\
                                                 .select(user_query_to_predict["user_id"],
                                                         user_query_to_predict["query_id"],
                                                         "user_id_index",
                                                         "query_id_index")

    user_query_to_predict = user_query_to_predict.repartition(8).cache()



    def fun2(x):
        out_put_list = []
        for q in b_query_id_list.value:
            if x[q] != None:
                out_put_list.append({
                    "user_id" : x["user_id"],
                    "query_id": q,
                    "rating"  : float(x[q])
                })
        return out_put_list

    user_query_rated_rdd = ut.rdd.flatMap(lambda x: fun2(x))
    user_query_rated = sc.createDataFrame(user_query_rated_rdd)

    user_query_rated = user_query_rated.join(user_set,  "user_id")\
                                       .join(query_set, "query_id")\
                                       .select(user_query_rated["user_id"],
                                               user_query_rated["query_id"],
                                               "user_id_index",
                                               "query_id_index",
                                               "rating")\
                                       .cache()


    (training, test) = user_query_rated.randomSplit([0.8, 0.2])

    als=ALS(maxIter=10,
        regParam=0.09,
        rank=25,
        userCol="user_id_index",
        itemCol="query_id_index",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True)

    model=als.fit(training)

    evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
    predictions=model.transform(test)
    rmse=evaluator.evaluate(predictions)
    print("RMSE="+str(rmse))


    predictions = sorted(model.transform(user_query_to_predict).collect(), key=lambda r: r[0])

    predicted = sc.createDataFrame(predictions)


    # predicted_df.write.options(header='True', delimiter=',') \
    #                .csv(data_path + "predicted_ALS")


    ## procedure to merge the utility_matrix with the predicted values
    predicted = predicted.drop("user_id_index", "query_id_index").withColumnRenamed("prediction", "predicted_rating")

    new_ut = predicted.groupBy(F.col("user_id")).pivot("query_id").avg("predicted_rating")
    new_ut = new_ut.repartition(8)

    ut = ut.alias("ut")
    new_ut = new_ut.alias("new_ut")

    query_id_common= set(new_ut.columns) - {"user_id"}
    query_id_other = set(ut.columns) - query_id_common - {"user_id"}

    ut = ut.join(new_ut, "user_id", how='left')
    ut = ut.repartition(20)
    ut = ut.select(
        [F.col("ut.user_id").alias("user_id")] +
        [(F.coalesce(F.col("ut."+x), F.col("new_ut."+x))).alias(x) for x in query_id_common] +
        [F.col("ut."+x).alias(x) for x in query_id_other]
    )


    with open(data_path + 'utility_matrix_filled_ALS.csv', 'w') as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(ut.columns)

        for row in ut.rdd.toLocalIterator():
            writer.writerow(row)

    end_session(sc)


def als_evaluate():
    print("Running als evaluation")

    sc = init_spark("query_recommendation")
    utility_matrix = load_utility_matrix(sc)

    ut = utility_matrix.repartition(8)

    query_set = load_query_set(sc)
    query_set = query_set.withColumn("query_id_index", F.monotonically_increasing_id())

    user_set = load_user_set(sc)
    user_set = user_set.select(F.col("id").alias("user_id")).withColumn("user_id_index", F.monotonically_increasing_id())

    query_user_masked = load_query_user_masked(sc)
    query_user_masked = query_user_masked.join(user_set,  'user_id')\
                                         .join(query_set, 'query_id')\
                                         .select(query_user_masked["user_id"],
                                                 query_user_masked["query_id"],
                                                 "user_id_index",
                                                 "query_id_index")

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

    user_query_to_predict = user_query_to_predict.join(user_set,  'user_id')\
                                                 .join(query_set, 'query_id')\
                                                 .select(user_query_to_predict["user_id"],
                                                         user_query_to_predict["query_id"],
                                                         "user_id_index",
                                                         "query_id_index")
    user_query_to_predict = user_query_to_predict.repartition(8).cache()
    user_query_to_predict = user_query_to_predict.unionByName(query_user_masked)


    def fun2(x):
        out_put_list = []
        for q in b_query_id_list.value:
            if x[q] != None:
                out_put_list.append({
                    "user_id" : x["user_id"],
                    "query_id": q,
                    "rating"  : float(x[q])
                })
        return out_put_list

    user_query_rated_rdd = ut.rdd.flatMap(lambda x: fun2(x))
    user_query_rated = sc.createDataFrame(user_query_rated_rdd)

    user_query_rated = user_query_rated.join(user_set,  "user_id")\
                                       .join(query_set, "query_id")\
                                       .select(user_query_rated["user_id"],
                                               user_query_rated["query_id"],
                                               "user_id_index",
                                               "query_id_index",
                                               "rating")\
                                       .cache()

    # take the rating of the masked
    query_user_rating_masked = user_query_rated.join(query_user_masked,
                                             (user_query_rated.query_id == query_user_masked.query_id) &
                                             (user_query_rated.user_id == query_user_masked.user_id),
                                             "inner").select(
                                                 user_query_rated.user_id.alias("user_id"),
                                                 user_query_rated.query_id.alias("query_id"),
                                                 user_query_rated.rating.alias("rating")
                                             )
    # subtract the masked
    user_query_rated = user_query_rated.join(query_user_masked,
                                             (user_query_rated.query_id == query_user_masked.query_id) &
                                             (user_query_rated.user_id == query_user_masked.user_id),
                                             "leftanti")
    (training, test) = user_query_rated.randomSplit([0.8, 0.2])

    als=ALS(maxIter=10,
        regParam=0.09,
        rank=25,
        userCol="user_id_index",
        itemCol="query_id_index",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True)

    model=als.fit(training)

    evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
    predictions=model.transform(test)
    rmse=evaluator.evaluate(predictions)
    print("model test RMSE="+str(rmse))


    predictions = sorted(model.transform(user_query_to_predict).collect(), key=lambda r: r[0])

    predicted = sc.createDataFrame(predictions)

    masked_predicted = predicted.join(query_user_rating_masked,
                                      (query_user_rating_masked["query_id"] == predicted["query_id"]) &
                                      (query_user_rating_masked.user_id == predicted.user_id),
                                      "inner").select(
                                                 predicted.user_id.alias("user_id"),
                                                 predicted.query_id.alias("query_id"),
                                                 query_user_rating_masked.rating.alias("rating"),
                                                 predicted.prediction.alias("prediction")
                                             )
    rmse=evaluator.evaluate(masked_predicted)
    print("masked RMSE="+str(rmse))
    ## RMSE=4.54686501389921
    end_session(sc)

if __name__ == "__main__":
    globals()[sys.argv[1]]()
