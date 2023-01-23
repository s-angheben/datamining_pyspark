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
    # rt = load_relational_table(sc)
    # tr = load_top_reviews_full_matrix(sc)
    # ut = load_utility_matrix(sc)
    # rt.createOrReplaceTempView("items")

    # # bu.printSchema()

    # print(check_consistency_rating(sc, ut, rt, tr))

    df = sc.read.option("header",True).options(inferSchema='True',delimiter=',').csv(data_path + "predicted")
    df.printSchema()
    print(df.count())


if __name__ == "__main__":
    test()
