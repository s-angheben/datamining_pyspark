from data_utils import *
from evaluate import compute_rmse
from utils import init_spark, end_session
from functools import reduce

from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator


def als():
    sc = init_spark("query_recommendation")
    utility_matrix = load_utility_matrix(sc)
    masked_ut, test_user_ids, test_query_ids = mask_utility_matrix(utility_matrix, 30, 100)

    # Store to file system
    save_masked_utility_matrix(masked_ut, test_user_ids, test_query_ids)

    ut = masked_ut.repartition(8)

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

    query_id_list = list(set(ut.columns) - {"user_id"})

    ut = ut.alias("ut")
    new_ut = new_ut.alias("new_ut")

    ut = ut.join(new_ut, "user_id", how='left')
    ut = ut.repartition(20)

    # ut.printSchema()

    ut = ut.select([F.col("ut.user_id").alias("user_id")] + [(F.coalesce(F.col("ut."+x), F.col("new_ut."+x))).alias(x) for x in query_id_list])

    # COMPUTE RMSE
    sliced_cols = ['user_id'] + test_query_ids

    # Compute RMSE evaluation
    slice_ut_original = utility_matrix.where(F.col('user_id').isin(test_user_ids)).select(sliced_cols)
    slice_ut_als = ut.where(F.col('user_id').isin(test_user_ids)).select(sliced_cols)
    rmse_ = compute_rmse(slice_ut_original, slice_ut_als)
    print('Computed RMSE for the ALS method is {}'.format(rmse_))

    # user_query_to_predict = user_query_to_predict.subtract(predicted.drop("predicted_rating"))

    with open(data_path + 'utility_matrix_filled_ALS.csv', 'w') as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(ut.columns)

        for row in ut.rdd.toLocalIterator():
            writer.writerow(row)

    end_session(sc)


if __name__ == "__main__":
    als()
