# Predicting Sales of Items in Shops

## Objective
Objective of this project is to develop a POC of an ML-based system for predicting sales of items in shops.
We will explore and evaluate the efficacy of different ML-methods to address this problem.

## Installation and Setup
1. Create a Python 3.11 based environment of your preference and install the requirements (`pip install -r requirements.txt`).
2. Adjust the rootpath and the names of raw data files for the 'local' environment inside: [config.yml](config.yml).

## Raw Data
Raw data comprise following tables and fields:
- shop_list: ['shop_name', 'shop_id]
- item_list: ['item_name', 'item_id', 'item_category_id']
- category_list: ['item_category_name', 'item_category_id']
- transaction: ['date', 'shop', 'item', 'price', 'amount' ]

The `Data` class provided by [data_utils.py](data_utils.py) module provides all the data upon instantiation. The schema of the transaction data is fixed ('shop', 'item' -> 'shop_id', 'item_id').
