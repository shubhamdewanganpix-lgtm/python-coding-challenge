# 🚀 PySpark Coding Challenge

## 📖 Overview
This repository contains a **PySpark data pipeline** that prepares training input data for a recommender system / click-through prediction model.

> ⚠️ The PyTorch model, embeddings, and training code are **not part of this repo**.  
> They will be implemented in a future iteration.  
> Our focus here is to:
> - Build training input data from raw behavioral logs (**clicks, carts, orders**).
> - Ensure sequences are constructed correctly (**last 1000 actions**).
> - Output the data in a format that can be consumed efficiently by **GPU training**.

---

## 🎯 High-Level Design of Training Inputs
Each record in the final training dataset corresponds to a **user impression** (an item shown to a user at a given time).

For each impression, we build the following fields:

| Column        | Type           | Description |
|---------------|----------------|-------------|
| `customer_id` | String         | Unique ID of the user |
| `dt`          | Timestamp/Date | Impression date |
| `actions`     | Array          | List of up to the last 1000 item IDs the user interacted with before the impression |
| `action_types`| Array          | Parallel array of action types (`click`, `cart`, `order`) |
| `impression_id` | String       | The item ID shown in the impression |
| `label`       | Integer (0/1)  | Target variable (1 if user later purchased the impression item, 0 otherwise) |

### Why this structure?
- **GPU-friendly**: actions and action types are in array form → directly mapped to embedding indices.  
- Each record is a **self-contained training example**, so PyTorch can batch them without extra joins.  
- Stored in **Parquet** → efficient columnar storage and fast I/O.  

---

## 🔮 How Training Will Work (Future Iteration)
- Item IDs will be mapped to **embedding indices** using a precomputed dictionary.
- Each `actions` sequence will be fed into a **sequential model** (e.g., Transformer, GRU, DeepFM).
- `action_types` embeddings may be concatenated to enrich the representation.
- The model will learn to predict the probability that the impression item (`impression_id`) will be clicked or ordered (`label`).
- Structured data enables **maximal GPU utilization** with minimal preprocessing.

---

## ⚙️ PySpark Pipeline

Implemented in **`data_pipeline.py`**

### Inputs
- `clicks_df`: user click logs  
- `carts_df`: user add-to-cart logs  
- `orders_df`: user order logs  
- `impressions_df`: impression logs (each row = user saw item at time `t`)

### Steps
1. Union clicks, carts, and orders into one `actions` DataFrame:  
   `(customer_id, item_id, action_type, ts)`
2. For each user, **sort actions by time** and keep only the **last 1000 actions** before each impression.
3. Join impressions with actions → build `(actions, action_types)` sequences.
4. Derive `label` by checking whether the `impression_id` was later ordered by the user.
5. Write output to **Parquet** for downstream training.

---

## ⚡ Performance Notes
- Spark **window functions** used for efficient ranking and filtering.  
- **Column pruning** reduces shuffle size.  
- **Parquet output** ensures fast I/O and columnar reads during GPU training.  

---

## 🧪 Tests

Implemented in **`tests/test_pipeline.py`** using **pytest**.

### ✅ Covered Scenarios
- **Small synthetic datasets**
  - Correctness of action ordering, labels, and sequence length.
- **Edge cases**
  - Users with no prior actions  
  - Users with only one action  
  - Users with >1000 actions (**trimmed**)  
  - Impressions with future actions (**filtered correctly**)  
  - Invalid timestamps handled gracefully
- **Large synthetic dataset**
  - Simulates production scale to test **performance and memory usage**.

Run tests locally:

```bash
pytest tests/


▶️ How to Run
On Local Spark
python pipeline.py \
  --clicks path/to/clicks.parquet \
  --carts path/to/carts.parquet \
  --orders path/to/orders.parquet \
  --impressions path/to/impressions.parquet \
  --output path/to/output/training_data.parquet


On Databricks

Upload the repo to a Databricks workspace and run:

from pipeline import build_training_dataset

training_df = build_training_dataset(clicks_df, carts_df, orders_df, impressions_df)
training_df.write.mode("overwrite").parquet("/mnt/output/training_data/")


📦 Deliverables in This Repo

README.md → project documentation (this file)
pipeline.py → PySpark pipeline implementation
tests/test_pipeline.py → unit & integration tests
synthetic_data_generator.py → generate fake click/cart/order/impression logs for testing
requirements.txt → Python dependencies
