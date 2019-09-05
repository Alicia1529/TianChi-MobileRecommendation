import pandas as pd
import numpy as np
from datetime import datetime
import pickle

filename = "Dec13_Dec18"
prediction_time = "2014-12-19"
prediction_time = datetime.strptime(prediction_time, '%Y-%m-%d')

item_file = "fresh_comp_offline/tianchi_fresh_comp_train_item.csv"
user_file = "data/{}.csv".format(filename)


items = pd.read_csv(item_file)
users = pd.read_csv(user_file).sort_values("time")

#商品本身的受欢迎度：收藏率，购买率， 添加购物车率 in percentage
#包括浏览、收藏、加购物车、购买，对应取值分别是1、2、3、4。
print("--item_table")
#convert time from string to datetime
users["time"] = users["time"].map(lambda x:datetime.strptime(x, '%Y-%m-%d %H') )

# views
item_table_view = users[users['behavior_type'] ==1 ].groupby("item_id")["item_id"].count().reset_index(name="item_total_views")
item_table_view["item_views_ratio"] = item_table_view["item_total_views"].div(item_table_view["item_total_views"].sum())

# save
item_table_save = users[users['behavior_type'] ==2 ].groupby("item_id")["item_id"].count().reset_index(name="item_total_saves")
item_table_save["item_saves_ratio"] = item_table_save["item_total_saves"].div(item_table_save["item_total_saves"].sum())

# shopping cart
item_table_shoppingcart = users[users['behavior_type'] ==3 ].groupby("item_id")["item_id"].count().reset_index(name="item_total_shoppingcarts")
item_table_shoppingcart["item_shoppingcarts_ratio"] = item_table_shoppingcart["item_total_shoppingcarts"].div(item_table_shoppingcart["item_total_shoppingcarts"].sum())

# purchase
item_table_purchase = users[users['behavior_type'] ==4 ].groupby("item_id")["item_id"].count().reset_index(name="item_total_purchases")
item_table_purchase["item_purchases_ratio"] = item_table_purchase["item_total_purchases"].div(item_table_purchase["item_total_purchases"].sum())

# 商品平均被浏览与购买的时间  对于每个user item pair => item 
# filter according to action_type, average purchase date- average view date, inner merge
user_item_first_purchase = users[users['behavior_type'] ==4 ].groupby(["user_id","item_id"])["time"].min().reset_index(name="user_item_first_purchase")
user_item_first_view = users[users['behavior_type'] ==1 ].groupby(["user_id","item_id"])["time"].min().reset_index(name="user_item_first_view")
user_item_gap_14 = pd.merge(user_item_first_purchase,user_item_first_view,how="inner",on=["user_id","item_id"])
user_item_gap_14["gap_14"] = user_item_gap_14["user_item_first_purchase"] - user_item_gap_14["user_item_first_view"]

user_item_gap_14["gap_14"] =user_item_gap_14["gap_14"].map(lambda x:x.days*24+x.seconds//3600)
#  convert original 0 to 0.5?
user_item_gap_14["gap_14"] =user_item_gap_14["gap_14"].apply(lambda x:0.5 if x==0 else x)
# get Reciprocal
user_item_gap_14["gap_14"] =user_item_gap_14["gap_14"].map(lambda x:7*24/x)
# if the value is negative, let it be 1
user_item_gap_14["gap_14"] =user_item_gap_14["gap_14"].apply(lambda x:1 if x<0 else x)
# if not appear, let it be 0(discussed later)
item_view_interval = user_item_gap_14.groupby("item_id")["gap_14"].mean().reset_index(name="item_view_interval")


# 商品平均被收藏与购买的时间  对于每个user item pair => item 
# filter according to action_type, average purchase date- average save date, inner merge
user_item_first_purchase = users[users['behavior_type'] ==4 ].groupby(["user_id","item_id"])["time"].min().reset_index(name="user_item_first_purchase")
user_item_first_save = users[users['behavior_type'] ==2 ].groupby(["user_id","item_id"])["time"].min().reset_index(name="user_item_first_save")
user_item_gap_24 = pd.merge(user_item_first_purchase,user_item_first_save,how="inner",on=["user_id","item_id"])
user_item_gap_24["gap_24"] = user_item_gap_24["user_item_first_purchase"] - user_item_gap_24["user_item_first_save"]
user_item_gap_24["gap_24"] = user_item_gap_24["gap_24"].map(lambda x:x.days*24+x.seconds//3600)
#  convert original 0 to 0.5?
user_item_gap_24["gap_24"] =user_item_gap_24["gap_24"].apply(lambda x:0.5 if x==0 else x)
# get Reciprocal
user_item_gap_24["gap_24"] =user_item_gap_24["gap_24"].map(lambda x:7*24/x)
# if the value is negative, let it be 1
user_item_gap_24["gap_24"] =user_item_gap_24["gap_24"].apply(lambda x:1 if x<0 else x)
# if not appear, let it be 0(discussed later)
item_save_interval = user_item_gap_24.groupby("item_id")["gap_24"].mean().reset_index(name="item_save_interval")



# 商品平均被添加购物车与购买的时间  对于每个user item pair => item 
# filter according to action_type, average purchase date- average save date, inner merge
user_item_first_purchase = users[users['behavior_type'] ==4 ].groupby(["user_id","item_id"])["time"].min().reset_index(name="user_item_first_purchase")
user_item_first_shoppingcart = users[users['behavior_type'] ==3 ].groupby(["user_id","item_id"])["time"].min().reset_index(name="user_item_first_shoppingcart")
user_item_gap_34 = pd.merge(user_item_first_purchase,user_item_first_shoppingcart,how="inner",on=["user_id","item_id"])
user_item_gap_34["gap_34"] = user_item_gap_34["user_item_first_purchase"] - user_item_gap_34["user_item_first_shoppingcart"]
user_item_gap_34["gap_34"] = user_item_gap_34["gap_34"].map(lambda x:x.days*24+x.seconds//3600)
#  convert original 0 to 0.5?
user_item_gap_34["gap_34"] =user_item_gap_34["gap_34"].apply(lambda x:0.5 if x==0 else x)
# get Reciprocal
user_item_gap_34["gap_34"] =user_item_gap_34["gap_34"].map(lambda x:7*24/x)
# if the value is negative, let it be 1
user_item_gap_34["gap_34"] =user_item_gap_34["gap_34"].apply(lambda x:1 if x<0 else x)
# if not appear, let it be 0(discussed later)
item_shoppingcart_interval = user_item_gap_34.groupby("item_id")["gap_34"].mean().reset_index(name="item_shoppingcart_interval")


# 商品购买人数占总浏览人数的比值
# 商品购买人数占总收藏人数的比值
# 商品购买人数占总加购物车人数的比值
item_num_purchase = users[users['behavior_type'] ==4 ].groupby(["item_id"])["item_id"].count().reset_index(name="item_num_purchase")
item_num_view = users[users['behavior_type'] ==1 ].groupby(["item_id"])["item_id"].count().reset_index(name="item_num_view")
item_num_save = users[users['behavior_type'] ==2 ].groupby(["item_id"])["item_id"].count().reset_index(name="item_num_save")
item_num_shoppingcart = users[users['behavior_type'] ==3 ].groupby(["item_id"])["item_id"].count().reset_index(name="item_num_shoppingcart")
item_percentage_table = pd.merge(item_num_purchase,item_num_view,on="item_id",how="outer").fillna(0)
item_percentage_table = pd.merge(item_percentage_table,item_num_save,on="item_id",how="outer").fillna(0)
item_percentage_table = pd.merge(item_percentage_table,item_num_shoppingcart,on="item_id",how="outer").fillna(0)
item_percentage_table["percent_41"] = item_percentage_table["item_num_purchase"]/item_percentage_table["item_num_view"]
item_percentage_table["percent_42"] = item_percentage_table["item_num_purchase"]/item_percentage_table["item_num_save"]
item_percentage_table["percent_43"] = item_percentage_table["item_num_purchase"]/item_percentage_table["item_num_shoppingcart"]
item_percentage_table = item_percentage_table.replace([np.inf, -np.inf], np.nan).fillna(0)[["item_id","percent_41","percent_42","percent_43"]]



# a table for all items 
item_table = pd.merge(item_table_view,item_table_save,on="item_id", how="outer").fillna(0)
item_table = pd.merge(item_table,item_table_shoppingcart,on="item_id", how="outer").fillna(0)
item_table = pd.merge(item_table,item_table_purchase,on="item_id", how="outer").fillna(0)
#---> if not appear, let it be 0(discussed later)
item_table = pd.merge(item_table,item_view_interval,on="item_id", how="outer").fillna(0)
item_table = pd.merge(item_table,item_save_interval,on="item_id", how="outer").fillna(0)
item_table = pd.merge(item_table,item_shoppingcart_interval,on="item_id", how="outer").fillna(0)
item_table = pd.merge(item_table,item_percentage_table, on="item_id", how="outer").fillna(0)
print(item_table.head())
path = "features/item_table_{}.pyc".format(filename)
pickle.dump(item_table, open(path,"wb"))


# user_item pair 
# 商品的购买频率：一般多久会购买一次-对同一用户而言
# never purchase again 0-> 24*50
# not accurate? not enough data?,standardize by num?
print("user,item - purchase frequency")

user_item_purchase_max = users[users['behavior_type'] ==4 ].groupby(["user_id","item_id"])["time"].max().reset_index(name="max")
user_item_purchase_min = users[users['behavior_type'] ==4 ].groupby(["user_id","item_id"])["time"].min().reset_index(name="min")
user_item_purchase_count = users[users['behavior_type'] ==4 ].groupby(["user_id","item_id"])["time"].count().reset_index(name="count")

user_item_table = pd.merge(user_item_purchase_max,user_item_purchase_min,on=["user_id","item_id"])
user_item_table = pd.merge(user_item_table,user_item_purchase_count,on=["user_id","item_id"])
user_item_table["UI_purchase_duration_hour"] = (user_item_table["max"]-user_item_table["min"])/user_item_table["count"]


user_item_table["UI_purchase_duration_hour"] = user_item_table["UI_purchase_duration_hour"].map(lambda x:x.days*24+x.seconds//3600)
user_item_table["UI_purchase_duration_hour"] = user_item_table["UI_purchase_duration_hour"].apply(lambda x: 24*50 if x == 0 else x)
user_item_table = user_item_table[["user_id","item_id","UI_purchase_duration_hour"]]

# 商品最后一次被浏览-预测时间的间隔
# 商品最后一次收藏-预测时间的间隔
# 商品最后一次添加购物册-预测时间的间隔
user_item_last_view = users[users['behavior_type'] ==1 ].groupby(["user_id","item_id"])["time"].max().reset_index(name="user_item_last_view")
user_item_last_view["user_item_last_view_predict"] = prediction_time-user_item_last_view["user_item_last_view"]
user_item_last_view["user_item_last_view_predict"] = user_item_last_view["user_item_last_view_predict"].apply(lambda x:x.days*24+x.seconds//3600)

user_item_last_save = users[users['behavior_type'] ==2 ].groupby(["user_id","item_id"])["time"].max().reset_index(name="user_item_last_save")
user_item_last_save["user_item_last_save_predict"] = prediction_time-user_item_last_save["user_item_last_save"]
user_item_last_save["user_item_last_save_predict"] = user_item_last_save["user_item_last_save_predict"].apply(lambda x:x.days*24+x.seconds//3600)
user_item_last_shoppingcart = users[users['behavior_type'] ==3 ].groupby(["user_id","item_id"])["time"].max().reset_index(name="user_item_last_shoppingcart")
user_item_last_shoppingcart["user_item_last_shoppingcart_predict"] = prediction_time-user_item_last_shoppingcart["user_item_last_shoppingcart"]
user_item_last_shoppingcart["user_item_last_shoppingcart_predict"] = user_item_last_shoppingcart["user_item_last_shoppingcart_predict"].apply(lambda x:x.days*24+x.seconds//3600)
# if add it to shopping cart one day before prediction day and didn't purchase
# first: put to shopping cart 
user_item_last_shoppingcart["user_item_shoppingcart_1_day"]  = user_item_last_shoppingcart["user_item_last_shoppingcart_predict"].apply(lambda x : 1 if x<24 and x>0 else 0 )

user_shoppingcart = user_item_last_shoppingcart[user_item_last_shoppingcart["user_item_shoppingcart_1_day"] ==1 ][["user_id","item_id"]]
# second: purchased at the last day 
user_purchased =  users[users['behavior_type'] ==4 ].groupby(["user_id","item_id"])["time"].max().reset_index(name="last_purchase_time")
user_purchased["user_item_last_day_purchase"] = prediction_time - user_purchased["last_purchase_time"]
user_purchased["user_item_last_day_purchase"] = user_purchased["user_item_last_day_purchase"].apply(lambda x: 1 if x.days*24+x.seconds//3600<24 and  x.days*24+x.seconds//3600> 0 else 0)
user_purchased = user_purchased[user_purchased["user_item_last_day_purchase"] ==1][["user_id","item_id"]]
ds1 = set([tuple(line) for line in user_shoppingcart.values])
ds2 = set([tuple(line) for line in user_purchased.values])
ds3 = ds1.difference(ds2) #shoppingcart&not purchase

shoppingcart_not_purchase = pd.DataFrame(list(ds1.difference(ds2)))
shoppingcart_not_purchase.columns =  ["user_id","item_id"]
shoppingcart_not_purchase["shoppingcart_notpurchase"] = 1

#fixme : fill value
user_item_table = pd.merge(user_item_table,user_item_last_view,on=["user_id", "item_id"], how="outer").fillna(0)
user_item_table = pd.merge(user_item_table,user_item_last_save,on=["user_id", "item_id"], how="outer").fillna(0)
user_item_table = pd.merge(user_item_table,user_item_last_shoppingcart,on=["user_id", "item_id"], how="outer").fillna(0)
user_item_table = pd.merge(user_item_table,shoppingcart_not_purchase,on=["user_id", "item_id"], how="outer").fillna(0)
user_item_table =user_item_table[["user_id","item_id","UI_purchase_duration_hour","user_item_last_view_predict","user_item_last_save_predict","user_item_last_shoppingcart_predict","shoppingcart_notpurchase"]]

path = "features/user_item_table_{}.pyc".format(filename)
pickle.dump(user_item_table, open(path,"wb"))


# 商品类别受欢迎程度：浏览，收藏率，购买率， 添加购物车率
# views
category_table_view = users[users['behavior_type'] ==1 ].groupby("item_category")["item_category"].count().reset_index(name="category_total_views")
category_table_view["category_views_ratio"] = category_table_view["category_total_views"].div(category_table_view["category_total_views"].sum())

# save
category_table_save = users[users['behavior_type'] ==2 ].groupby("item_category")["item_category"].count().reset_index(name="category_total_saves")
category_table_save["category_saves_ratio"] = category_table_save["category_total_saves"].div(category_table_save["category_total_saves"].sum())

# shopping cart
category_table_shoppingcart = users[users['behavior_type'] ==3 ].groupby("item_category")["item_category"].count().reset_index(name="category_total_shoppingcarts")
category_table_shoppingcart["category_shoppingcarts_ratio"] = category_table_shoppingcart["category_total_shoppingcarts"].div(category_table_shoppingcart["category_total_shoppingcarts"].sum())

# purchase
category_table_purchase = users[users['behavior_type'] ==4 ].groupby("item_category")["item_category"].count().reset_index(name="category_total_purchases")
category_table_purchase['category_purchases_ratio'] = category_table_purchase["category_total_purchases"].div(category_table_purchase["category_total_purchases"].sum())

# a table for all categories 
category_table = pd.merge(category_table_view,category_table_save,on="item_category", how="outer").fillna(0)
category_table = pd.merge(category_table,category_table_shoppingcart,on="item_category", how="outer").fillna(0)
category_table = pd.merge(category_table,category_table_purchase,on="item_category", how="outer").fillna(0)

path = "features/category_table_{}.pyc".format(filename)
pickle.dump(category_table, open(path,"wb"))

# user_category pair 
# 商品类别购买频率：一般多久会购买一次-对同一用户而言
# never purchase again 0-> 24*50
# not accurate? not enough data?,standardize by num?
print("user,category - purchase frequency")
user_category_purchase_max = users[users['behavior_type'] ==4 ].groupby(["user_id","item_category"])["time"].max().reset_index(name="max")
user_category_purchase_min = users[users['behavior_type'] ==4 ].groupby(["user_id","item_category"])["time"].min().reset_index(name="min")
user_category_purchase_count = users[users['behavior_type'] ==4 ].groupby(["user_id","item_category"])["time"].count().reset_index(name="count")

user_category_table = pd.merge(user_category_purchase_max,user_category_purchase_min,on=["user_id","item_category"])
user_category_table = pd.merge(user_category_table,user_category_purchase_count,on=["user_id","item_category"])
user_category_table["UC_purchase_duration_hour"] = (user_category_table["max"]-user_category_table["min"])/user_category_table["count"]


user_category_table["UC_purchase_duration_hour"] = user_category_table["UC_purchase_duration_hour"].map(lambda x:x.days*24+x.seconds//3600)
user_category_table["UC_purchase_duration_hour"] = user_category_table["UC_purchase_duration_hour"].apply(lambda x: 24*50 if x == 0 else x)
user_category_table = user_category_table.sort_values("UC_purchase_duration_hour",ascending=True)

user_category_table = user_category_table[["user_id","item_category","UC_purchase_duration_hour"]]
# 类别最后一次被浏览-预测时间的间隔
# 类别最后一次收藏-预测时间的间隔
# 类别最后一次添加购物册-预测时间的间隔

user_category_last_view = users[users['behavior_type'] ==1 ].groupby(["user_id","item_category"])["time"].max().reset_index(name="user_category_last_view")
user_category_last_view["user_category_last_view_predict"] = prediction_time-user_category_last_view["user_category_last_view"]
user_category_last_view["user_category_last_view_predict"] = user_category_last_view["user_category_last_view_predict"].apply(lambda x:x.days*24+x.seconds//3600)

user_category_last_save = users[users['behavior_type'] ==2 ].groupby(["user_id","item_category"])["time"].max().reset_index(name="user_category_last_save")
user_category_last_save["user_category_last_save_predict"] = prediction_time-user_category_last_save["user_category_last_save"]
user_category_last_save["user_category_last_save_predict"] = user_category_last_save["user_category_last_save_predict"].apply(lambda x:x.days*24+x.seconds//3600)


user_category_last_shoppingcart = users[users['behavior_type'] ==3 ].groupby(["user_id","item_category"])["time"].max().reset_index(name="user_category_last_shoppingcart")
user_category_last_shoppingcart["user_category_last_shoppingcart_predict"] = prediction_time-user_category_last_shoppingcart["user_category_last_shoppingcart"]
user_category_last_shoppingcart["user_category_last_shoppingcart_predict"] = user_category_last_shoppingcart["user_category_last_shoppingcart_predict"].apply(lambda x:x.days*24+x.seconds//3600)

#fixme : fill value
user_category_table = pd.merge(user_category_table,user_category_last_view,on=["user_id", "item_category"], how="outer").fillna(0)
user_category_table = pd.merge(user_category_table,user_category_last_save,on=["user_id", "item_category"], how="outer").fillna(0)
user_category_table = pd.merge(user_category_table,user_category_last_shoppingcart,on=["user_id", "item_category"], how="outer").fillna(0)
user_category_table =user_category_table[["user_id","item_category","UC_purchase_duration_hour","user_category_last_view_predict","user_category_last_save_predict","user_category_last_shoppingcart_predict"]]



path = "features/user_category_table_{}.pyc".format(filename)
pickle.dump(user_category_table, open(path,"wb"))
