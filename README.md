# Predicting Sales of Items in Shops

## Objective
Objective of this project is to develop a POC of an ML-based system for predicting sales of items in shops.
We will explore and evaluate the efficacy of different ML-methods to address this problem.

## Installation and Setup
1. Create a Python 3.11 based environment of your preference and install the repository as a package (`pip install -e .`).
2. Adjust the rootpath and the names of raw data files for the 'local' environment inside: [config.yml](config/config.yml).

## Data

### Raw Data
Raw data comprise following tables and fields:
- shop_list: ['shop_name', 'shop_id]
- item_list: ['item_name', 'item_id', 'item_category_id']
- category_list: ['item_category_name', 'item_category_id']
- transaction: ['date', 'shop', 'item', 'price', 'amount' ]

The `Data` class provided by [data_utils.py](src/data_utils.py) module loads the raw data upon instantiation, after fixing some data schema issues (in transactions table, 'shop', 'item' -> 'shop_id', 'item_id'). 

### Exploratory Data Analysis
See [EDA.ipynb](notebooks/EDA.ipynb). To ease data analysis and interpretability, the 4 tables are merged into one single table , date strings are converted to datetime objects, and the implausible (negative) and outliers foundnd in price and amount data are cleaned. All these functions are provided by the `Data` class. After the data processing, the temporal trends in transaction counts and total sale amounts are analyzed, revealing both a yearly trend and a seasonality.

