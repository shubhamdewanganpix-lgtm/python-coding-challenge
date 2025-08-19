import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.utils import preprocess_actions, generate_training_data

def main(start_date, end_date, output_path):
    spark = SparkSession.builder.appName("PySparkCodingChallenge").getOrCreate()

    # Example: replace with real Kafka / DW reads
    clicks_df = spark.read.parquet("input/clicks/")
    carts_df = spark.read.parquet("input/add_to_carts/")
    orders_df = spark.read.parquet("input/orders/")
    impressions_df = spark.read.parquet("input/impressions/")

    # Preprocess raw actions
    actions_df = preprocess_actions(clicks_df, carts_df, orders_df)

    # Generate training input
    training_df = generate_training_data(impressions_df, actions_df)

    # Write partitioned output
    (training_df
        .write
        .mode("overwrite")
        .partitionBy("dt")
        .parquet(output_path))

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    main(args.start_date, args.end_date, args.output_path)
