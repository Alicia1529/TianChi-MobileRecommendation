import pickle
import pandas as pd
import numpy as np

user_table_file = "features/u_df.pyc"
item_table_file = "features/item_table.pyc"
user_item = "features/ui_df.pyc"
user_category = "features/uc_df.py"
user_category_table_file = "features/user_category_frequency.pyc"
user_item_table_file = "features/user_item_frequency.pyc"

user_table = pickle.load(open(user_table_file,"rb"))
user_item = pickle.load(open(user_item,"rb"))
user_category = pickle.load(open(user_category,"rb"))
user_item_table = pickle.load(open(user_item_table_file,"rb"))
item_table = pickle.load(open(item_table_file,"rb"))
user_category_table= pickle.load(open(user_category_table_file,"rb"))
print(user_item_table.columns)
print(user_item_table.head())
# print(item_table.columns)
# print(user_item_table.head())
# print(item_table.head())

#user_item_table contains unique user, item pairs 
full_table = pd.merge(user_item_table, item_table, how="left", on =["item_id"])
print(full_table.head())
# # full_table = pd.merge(full_table, user_category_table, how="left", on =["user_id","item_category"])


print(full_table.columns)
# print(full_table.count())
# train_table = full_table.sample(frac=0.1)
