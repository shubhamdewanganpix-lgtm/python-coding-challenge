import pytest
from pyspark.sql import SparkSession
from src.utils import preprocess_actions, generate_training_data

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.master("local[*]").appName("pytest").getOrCreate()

def test_empty_actions(spark):
    clicks = spark.createDataFrame([], "customer_id INT, item_id INT, click_time TIMESTAMP")
    carts = spark.createDataFrame([], "customer_id INT, config_id INT, simple_id INT, occurred_at TIMESTAMP")
    orders = spark.createDataFrame([], "order_date DATE, customer_id INT, config_id INT, simple_id INT, occurred_at TIMESTAMP")

    actions = preprocess_actions(clicks, carts, orders)
    assert actions.count() == 0

def test_action_ranking(spark):
    impressions = spark.createDataFrame(
        [(1, 100, "2023-01-05")], "customer_id INT, item_id INT, dt STRING"
    )
    clicks = spark.createDataFrame(
        [(1, 101, "2023-01-01"), (1, 102, "2023-01-02")],
        "customer_id INT, item_id INT, click_time STRING"
    )
    carts = spark.createDataFrame([], "customer_id INT, config_id INT, simple_id INT, occurred_at STRING")
    orders = spark.createDataFrame([], "order_date STRING, customer_id INT, config_id INT, simple_id INT, occurred_at STRING")

    actions = preprocess_actions(clicks, carts, orders)
    training = generate_training_data(impressions, actions)

    row = training.collect()[0]
    assert row["impression"] == 100
    assert len(row["actions"]) <= 1000

def test_large_dataset(spark):
    impressions = spark.range(10000).withColumnRenamed("id", "customer_id") \
        .withColumn("item_id", (F.col("customer_id") % 100)) \
        .withColumn("dt", F.lit("2023-01-10"))
    clicks = spark.range(100000).withColumnRenamed("id", "customer_id") \
        .withColumn("item_id", (F.col("customer_id") % 50)) \
        .withColumn("click_time", F.lit("2023-01-01"))

    carts = spark.createDataFrame([], "customer_id INT, config_id INT, simple_id INT, occurred_at STRING")
    orders = spark.createDataFrame([], "order_date STRING, customer_id INT, config_id INT, simple_id INT, occurred_at STRING")

    actions = preprocess_actions(clicks, carts, orders)
    training = generate_training_data(impressions, actions)

    assert training.count() > 0
