## LOAD THE business-users matrix
## Compare the predicted rate of a query with the "true" value
##

from utils import *
from data_utils import *


def load_top_reviews_full_matrix(sc):
    business_user = sc.read.option("header", True)\
                           .option("inferSchema", True)\
                           .csv(data_path + 'top_reviews.csv')
    return business_user


# TODO, NOT FINISHED
def get_true_query_rating(sc, queryRow, user_id, relational_table, top_reviews):
    # business_user.createOrReplaceTempView("topreviews")

    rs = get_query_result_set_id(sc, queryRow.query_string, relational_table)

    business_id_list = rs.rdd.map(lambda x: x.business_id).collect()

    subset = top_reviews.filter(col("business_id").isin(business_id_list))\
                        .filter(col("user_id") == user_id)

    count = subset.count()
    subset.select(sum(subset.rating)).show()


    print(count)
    # print(rating_sum)

def test():
    sc = init_spark("evaluation")
    rt = load_relational_table(sc)
    tr = load_top_reviews_full_matrix(sc)
    # bu.printSchema()
    query1 = load_query_set(sc).first()
    print(query1)

    get_true_query_rating(sc, query1, "WVqS85AUR20gbSFkuKH8Ig", rt, tr)


if __name__ == "__main__":
    test()
