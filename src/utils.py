from pyspark.sql import functions as F, Window

def preprocess_actions(clicks_df, carts_df, orders_df):
    # Normalize schema
    clicks = clicks_df.withColumn("action_type", F.lit(1)) \
        .withColumnRenamed("item_id", "config_id") \
        .withColumnRenamed("click_time", "ts")

    carts = carts_df.withColumn("action_type", F.lit(2)) \
        .withColumnRenamed("occurred_at", "ts")

    orders = orders_df.withColumn("action_type", F.lit(3)) \
        .withColumnRenamed("occurred_at", "ts")

    return clicks.select("customer_id", "config_id", "ts", "action_type") \
        .unionByName(carts.select("customer_id", "config_id", "ts", "action_type")) \
        .unionByName(orders.select("customer_id", "config_id", "ts", "action_type"))

def generate_training_data(impressions_df, actions_df):
    # Join impressions with actions
    joined = impressions_df.alias("imp") \
        .join(actions_df.alias("act"),
              (F.col("imp.customer_id") == F.col("act.customer_id")) &
              (F.col("act.ts") < F.col("imp.dt")),
              "left")

    # Rank actions per customer up to 1000
    w = Window.partitionBy("imp.customer_id", "imp.dt").orderBy(F.col("act.ts").desc())
    ranked = joined.withColumn("rank", F.row_number().over(w)) \
                   .filter(F.col("rank") <= 1000)

    # Aggregate to arrays
    result = ranked.groupBy("imp.customer_id", "imp.item_id", "imp.dt") \
        .agg(
            F.collect_list("act.config_id").alias("actions"),
            F.collect_list("act.action_type").alias("action_types")
        )

    return result.withColumnRenamed("item_id", "impression")
