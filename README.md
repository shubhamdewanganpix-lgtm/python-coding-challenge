Overview

This repository contains a PySpark data pipeline that prepares training input data for a recommender system / click-through prediction model.

The PyTorch model, embeddings, and training code are not part of this repo (they will be implemented later). Our goal here is to:

Build training input data from raw behavioral logs (clicks, carts, orders).

Ensure sequences are constructed correctly (last 1000 actions).

Output the data in a format that can be consumed efficiently by GPU training.

High-Level Design of Training Inputs

Each record in the final training dataset corresponds to a user impression (an item shown to a user at a given time).

For each impression, we build the following fields:

Column	Type	Description
customer_id	String	Unique ID of the user
dt	Timestamp/Date	Impression date
actions	Array<String>	List of up to the last 1000 item IDs the user interacted with before the impression
action_types	Array<String>	Parallel array of action types (click, cart, order)
impression_id	String	The item ID shown in the impression
label	Integer (0/1)	Target variable (1 if user later purchased the impression item, 0 otherwise)

This structure is GPU-friendly:

Actions and action types are already in array form → can be mapped to embedding indices directly.

Each record is a self-contained training example, so PyTorch can batch them without extra joins.

Stored in Parquet, which provides efficient columnar storage and fast I/O.

How Training Will Work (future iteration)

Item IDs will be mapped to embedding indices using a precomputed dictionary.

Each actions sequence will be fed into a sequential model (e.g., Transformer, GRU, or DeepFM).

action_types embeddings may be concatenated to enrich the representation.

The model will learn to predict the probability that the impression item (impression_id) will be clicked or ordered (label).

Since data is structured efficiently, GPU training can be maximized with minimal preprocessing.

PySpark Pipeline

Implemented in pipeline.py:

Inputs

clicks_df: user click logs

carts_df: user add-to-cart logs

orders_df: user order logs

impressions_df: impression logs (each row = user saw item at time t)

Steps

Union clicks, carts, orders into one actions DataFrame with schema (customer_id, item_id, action_type, ts)

For each user, sort actions by time and keep only the last 1000 actions before each impression.

Join impressions with actions → build (actions, action_types) sequences.

Derive label by checking whether the impression_id was later ordered by the user.

Write output to Parquet for downstream training.

Performance Notes

Uses Spark window functions to rank/filter actions efficiently.

Column pruning to reduce shuffle size.

Parquet output ensures efficient I/O and columnar reads during GPU training.

Tests

Implemented in tests/test_pipeline.py using pytest:

✅ Small synthetic datasets (correctness of action ordering, labels, sequence length).

✅ Edge cases:

Users with no prior actions

Users with only one action

Users with >1000 actions (trimmed to last 1000)

Impressions with future actions (filtered correctly)

Invalid timestamps handled

✅ Large synthetic dataset to simulate production scale (performance + memory usage).

Run tests locally:

pytest tests/

How to Run
1. On Local Spark
python pipeline.py \
  --clicks path/to/clicks.parquet \
  --carts path/to/carts.parquet \
  --orders path/to/orders.parquet \
  --impressions path/to/impressions.parquet \
  --output path/to/output/training_data.parquet

2. On Databricks

Upload the repo to a Databricks workspace and run:

from pipeline import build_training_dataset

training_df = build_training_dataset(clicks_df, carts_df, orders_df, impressions_df)
training_df.write.mode("overwrite").parquet("/mnt/output/training_data/")

Deliverables in This Repo

README.md (this file)

pipeline.py → PySpark pipeline implementation

tests/test_pipeline.py → Unit & integration tests

synthetic_data_generator.py → Create fake click/cart/order/impression logs for testing

requirements.txt → Python dependencies