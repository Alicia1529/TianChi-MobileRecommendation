import pickle
import pandas as pd
import numpy as np


# label_data_name = "Dec13_Dec18"
# feature_data_name = "Dec13_Dec18"
# output_data_name = "Dec13_Dec18"

# label_data_name = "Dec3"
# feature_data_name = "Nov26_Dec2"
# output_data_name = "Nov26_Dec3"

label_data_name = "Nov25"
feature_data_name = "Nov18_Nov24"
output_data_name = "Nov18_Nov25"

al_user_table_file = "features/u_df_{}.pyc".format(feature_data_name)
al_user_item = "features/ui_df_{}.pyc".format(feature_data_name)
al_user_category = "features/uc_df_{}.pyc".format(feature_data_name)
#####item#####alicia
tl_item_table_file = "features/item_table_{}.pyc".format(feature_data_name)
tl_category_table_file = "features/category_table_{}.pyc".format(feature_data_name)
tl_user_item_table_file = "features/user_item_table_{}.pyc".format(feature_data_name)
tl_user_category_table_file = "features/user_category_table_{}.pyc".format(feature_data_name)

al_user_table = pickle.load(open(al_user_table_file,"rb"))
print("al_user_table.columns")
print(al_user_table.columns)

al_user_item_table = pickle.load(open(al_user_item,"rb"))
print("al_user_item_table.columns")
print(al_user_item_table.columns)

al_user_category_table = pickle.load(open(al_user_category,"rb"))
print("al_user_category_table.columns ")
print(al_user_category_table.columns )
#####item####alicia
tl_item_table = pickle.load(open(tl_item_table_file,"rb"))
print(tl_item_table.columns)
print("tl_item_table.columns")
tl_category_table = pickle.load(open(tl_category_table_file,"rb"))
print(tl_category_table.columns)
print("tl_category_table.columns")
tl_user_item_table= pickle.load(open(tl_user_item_table_file,"rb"))
print(tl_user_item_table.columns)
print("tl_user_item_table.columns")
tl_user_category_table= pickle.load(open(tl_user_category_table_file,"rb"))
print(tl_user_category_table.columns)
print("tl_user_category_table.columns")


#get labeled data ###############
filename = "features/{}_label.pyc".format(label_data_name)
user_item_label = pickle.load(open(filename,"rb"))
item_info = pd.read_csv("fresh_comp_offline/tianchi_fresh_comp_train_item.csv")[["item_id","item_category"]]

full_df = pd.merge(user_item_label,item_info,how = "left",on=["item_id"])
#user_id, item_id, label, item category
full_df = pd.merge(full_df,al_user_table,how="left",on=["user_id"])
full_df = pd.merge(full_df,al_user_item_table,how="left",on=["user_id","item_id"])
full_df = pd.merge(full_df,al_user_category_table,how="left",on=["user_id","item_category"])
full_df = pd.merge(full_df,tl_item_table,how="left",on=["item_id",])
full_df = pd.merge(full_df,tl_category_table,how="left",on=["item_category"])
full_df = pd.merge(full_df,tl_user_item_table,how="left",on=["user_id","item_id"])
full_df = pd.merge(full_df,tl_user_category_table,how="left",on=["user_id","item_category"])

print("full_table.columns")
print(full_df.columns)
print("full_table.head()")
print(full_df.head())
filename = "data/train_df_{}".format(output_data_name)
pickle.dump(full_df,open(filename,"wb"))