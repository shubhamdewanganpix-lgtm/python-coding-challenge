#!/bin/bash
set -e

echo "Running PySpark pipeline for 14 days of training data..."

spark-submit \
  --master local[*] \
  --driver-memory 8g \
  --executor-memory 4g \
  src/data_pipeline.py \
  --start-date 2023-01-01 \
  --end-date 2023-01-14 \
  --output-path output/

echo "Pipeline finished. Data stored in ./output/"
