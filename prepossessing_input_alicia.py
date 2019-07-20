import pickle
import pandas as pd
import numpy as np

tl_item_table_file = "features/item_table.pyc"
tl_category_table_file = "features/category_table.pyc"
tl_user_item_table_file = "features/user_item_frequency.pyc"
tl_user_category_table_file = "features/user_category_frequency.pyc"


tl_item_table = pickle.load(open(tl_item_table_file,"rb"))
tl_category_table = pickle.load(open(tl_category_table_file,"rb"))
tl_user_item_table= pickle.load(open(tl_user_item_table_file,"rb"))
tl_user_category_table= pickle.load(open(tl_user_category_table_file,"rb"))
print(tl_item_table.columns)
print(tl_item_table.head())
print("--------------")
print(tl_category_table.columns)
print(tl_category_table.head())
print("--------------")
print(tl_user_item_table.columns)
print(tl_user_item_table.head())
print("--------------")
print(tl_user_category_table.columns)
print(tl_user_category_table.head())

# #user_item_table contains unique user, item pairs 
# full_table = pd.merge(user_item_table, item_table, how="left", on =["item_id"])
# print(full_table.head())
# # # full_table = pd.merge(full_table, user_category_table, how="left", on =["user_id","item_category"])


# print(full_table.columns)
# # print(full_table.count())
# # train_table = full_table.sample(frac=0.1)
