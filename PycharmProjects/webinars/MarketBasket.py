
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from mlxtend.preprocessing import TransactionEncoder

# Set default asthetic parameters.
sns.set()

# Define path to data.
data_path = 'https://github.com/datacamp/Market-Basket-Analysis-in-python-live-training/raw/master/data/'
pd.set_option('display.max_columns', None)

orders = pd.read_csv(data_path + 'olist_order_items_dataset.csv' )
products = pd.read_csv(data_path + 'olist_products_dataset.csv' )
translations = pd.read_csv(data_path + 'product_category_name_translation.csv' )


#print(orders.info())
print("new")
products = products.merge(translations, on='product_category_name', how='left')


orders = orders.merge(products, on='product_id', how='left')
print("new")
#print(orders.head())
orders.dropna(inplace=True, subset=['product_category_name_english'])
#print(orders.product_id.nunique())
#print(orders.product_category_name_english.nunique())
grouped = orders.groupby('order_id')['product_category_name_english'].unique()
#Used to get unique values in each transaction
transactions = grouped.tolist()
#print(len(transactions))
#counts = [len(transaction) for transaction in transactions]
#print(Counter(counts))
#The idea of all of this is to get the transactions into a list of lists
# where each list in the list is a transactions



encoder = TransactionEncoder()
encoder.fit(transactions)
onehot = encoder.transform(transactions)
onehot = pd.DataFrame(onehot,columns=encoder.columns_)

'''support = onehot.mean().sort_values(ascending=False)
print(support.head())
print(onehot.sum(axis=1).value_counts())

onehot['sports_leisure_health_beauty'] = onehot['sports_leisure'] & onehot['health_beauty']
print('support  ', onehot['sports_leisure_health_beauty'].mean())

# Merge books_imported and books_technical.
onehot['books'] = onehot['books_imported'] | onehot['books_technical']

# Print support values for books, books_imported, and books_technical.
print(onehot[['books','books_imported','books_technical']].mean(axis=0))
'''

#The Apriori Algorithm and Pruning¶

from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(onehot, min_support=.00005, use_colnames=True)
#print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric='leverage', min_threshold=.0)
print(rules)


'''
import pandas as pd Imports the pandas package with the alias pd
.head() Prints the header of a DataFrame
.dtypes Gets the data types of each column in a
DataFrame
.info() Returns a # observations, data types, and
missing values per column
.describe() Returns statistical distribution of numeric
value columns in a DataFrame
.merge() Performs a database-style join on DataFrame
objects
.unique() Returns the unique rows in a DataFrame or
entries in a column
.groupby() Splits a DataFrame into multiple objects
according to some criterion
.plot(kind=‘bar’) Visualizes barplot
.tolist() Converts DataFrame into list
Functions Description
TransactionEncoder() Instantiates a transaction encoder
.fit() Fits instance of encoder to transaction data
.transform() Transforms list of lists into one-hot encoded
array of transactions
.mean(axis=0) Computes support values from one-hot
encoded data
apriori() Identifies frequent itemsets from one-hot
encoded transactions
association_rules() Computes association rules and prunes them
according to metric thresholds
sns.scatterplot() Generates scatterplot of two variables
from mlxtend import
preprocessing
Imports the preprocessing module from
mlxtend
from mlxtend import
frequent_patterns
Imports the frequent_patterns module from
mlxtend
.dropna() Drops missing values from DataFrame

''' #functions used