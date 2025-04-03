# Flight Booking Nature Language to SQL

## Execution Instructions
Best results commands:
```
python3.10 train_t5.py --finetune --learning_rate 0.001
python3.10 train_t5.py
python3.10 prompting.py --shot 3
```

## Environment

It's highly recommended to use a virtual environment (e.g. conda, venv) for this assignment.

Example of virtual environment creation using conda:
```
conda create -n env_name python=3.10
conda activate env_name
python -m pip install -r requirements.txt
```

## Evaluation commands

If you have saved predicted SQL queries and associated database records, you can compute F1 scores using:
```
python evaluate.py
  --predicted_sql results/t5_ft_dev.sql
  --predicted_records records/t5_ft_dev.pkl
  --development_sql data/dev.sql
  --development_records records/ground_truth_dev.pkl
```
