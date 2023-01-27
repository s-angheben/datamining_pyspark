## LOAD THE business-users matrix
## Compare the predicted rate of a query with the "true" value
##

from utils import *
from data_utils import *
import math

def load_top_reviews_full_matrix(sc):
    business_user = sc.read.option("header", True)\
                           .option("inferSchema", True)\
                           .csv(data_path + 'top_reviews.csv')
    return business_user


# MAYBE NEED OPTIMIZATIONS
def get_true_query_rating(sc, queryRow, user_id, relational_table, top_reviews):
    # business_user.createOrReplaceTempView("topreviews")

    rs = get_query_result_set_id(sc, queryRow.query_string)

    business_id_list = rs.rdd.map(lambda x: x.business_id).collect()

    subset = top_reviews.filter(F.col("business_id").isin(business_id_list))\
                        .filter(F.col("user_id") == user_id)

    rating_count = subset.count()
    rating_sum = subset.select(F.sum(subset.rating)).collect()[0][0]

    true_rating = round(float(rating_sum) * 100.0 / (rating_count * 5.0))
    if true_rating > 100:
        true_rating = 100

    return true_rating



def select_random_query_user(df, n):
    df = df.orderBy(F.rand()).limit(n).select("query_id", "user_id")
    return df


## calculate the true_query_rating on a query and user for which we already
## have the rating in the utility_matrix
## check if we get the same result
def check_consistency_rating(sc, ut, rt, tr):
    query1 = load_query_set(sc).first()
    print(query1)

    a = ut.filter(F.col(query1.query_id).isNotNull()).select(F.col("user_id"), F.col(query1.query_id)).first()
    print(a)
    user_id = a.__getitem__('user_id')
    rating  = int(a.__getitem__(query1.query_id))
    # print(user_id)
    # print(rating)

    true_rating = get_true_query_rating(sc, query1, user_id, rt, tr)
    print(true_rating)
    return true_rating == rating


def test():
    sc = init_spark("evaluation")
    ut = load_utility_matrix(sc)
    ut.createOrReplaceTempView("utility_matrix")

    relational_table = load_relational_table(sc)
    relational_table.createOrReplaceTempView("items")

    query_set = load_query_set(sc)
    query_set = query_set.withColumn("query_id_index", F.monotonically_increasing_id())

    user_set = load_user_set(sc)
    user_set = user_set.select(F.col("id").alias("user_id")).withColumn("user_id_index", F.monotonically_increasing_id())


    query_id_list = set(ut.columns) - {"user_id"}
    b_query_id_list = sc.sparkContext.broadcast(query_id_list)


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

    #print(user_query_rated.count()) ## 113981
    df = select_random_query_user(user_query_rated, 10000)
    save_random_query_user(df)

if __name__ == "__main__":
    test()
