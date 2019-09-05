import pandas as pd
import numpy as np
from datetime import datetime
import pickle

# date_range = "Nov18_Nov24"
date_range = "Dec13_Dec18"

user_path = "./data/{}.csv".format(date_range)
user_df = pd.read_csv(user_path)
user_df["time"] = user_df["time"].map(lambda x:datetime.strptime(x, '%Y-%m-%d %H') )
# print(user_df.count()[0]) # 23291027

### ui
# 每个用户商品对对应的4个量及总量
user_df['ui_cumcount'] = user_df.groupby(['user_id','item_id','behavior_type']).cumcount()
ui_df_bcount = user_df.drop_duplicates(['user_id','item_id','behavior_type'], \
    'last')[['user_id','item_id','behavior_type','ui_cumcount']]
ui_df_bcount = pd.get_dummies(ui_df_bcount['behavior_type']).join(ui_df_bcount[['user_id','item_id','ui_cumcount']])
ui_df_bcount.rename(columns = \
    {1:'behavior_type_1', 2:'behavior_type_2', 3:'behavior_type_3',4:'behavior_type_4'}, inplace=True)
ui_df_bcount['ui_b1count'] = ui_df_bcount['behavior_type_1'] * (ui_df_bcount['ui_cumcount']+1)
ui_df_bcount['ui_b2count'] = ui_df_bcount['behavior_type_2'] * (ui_df_bcount['ui_cumcount']+1)
ui_df_bcount['ui_b3count'] = ui_df_bcount['behavior_type_3'] * (ui_df_bcount['ui_cumcount']+1)
ui_df_bcount['ui_b4count'] = ui_df_bcount['behavior_type_4'] * (ui_df_bcount['ui_cumcount']+1)
ui_df_bcount = ui_df_bcount.groupby(['user_id', 'item_id']) \
    .agg({'ui_b1count':np.sum, 'ui_b2count':np.sum, 'ui_b3count':np.sum, 'ui_b4count':np.sum})
ui_df_bcount.reset_index(inplace=True)

ui_df_avg = ui_df_bcount.groupby(['user_id']) \
    .agg({'ui_b1count':['mean','sum'], 'ui_b2count':['mean','sum'], 'ui_b3count':['mean','sum'], 'ui_b4count':['mean','sum']})
ui_df_avg.columns = ['_'.join(col).strip() for col in ui_df_avg.columns.values]
ui_df_avg.reset_index(inplace=True)

# ui_df
ui_df = pd.merge(ui_df_bcount, ui_df_avg, how='left', on=['user_id'])
ui_df['ui_b1count_mean'] = (ui_df['ui_b1count']-ui_df['ui_b1count_mean'])/ui_df['ui_b1count_sum']
ui_df['ui_b2count_mean'] = (ui_df['ui_b2count']-ui_df['ui_b2count_mean'])/ui_df['ui_b2count_sum']
ui_df['ui_b3count_mean'] = (ui_df['ui_b3count']-ui_df['ui_b3count_mean'])/ui_df['ui_b3count_sum']
ui_df['ui_b4count_mean'] = (ui_df['ui_b4count']-ui_df['ui_b4count_mean'])/ui_df['ui_b4count_sum']
ui_df = ui_df[['user_id','item_id','ui_b1count_mean','ui_b2count_mean','ui_b3count_mean','ui_b4count_mean']]
print(ui_df.head())
pickle.dump(ui_df, open("./features/ui_df_{}.pyc".format(date_range),"wb"))

### uc
# 每个用户商品类别对对应的4个量及总量
user_df['uc_cumcount'] = user_df.groupby(['user_id','item_category','behavior_type']).cumcount()
uc_df_bcount = user_df.drop_duplicates(['user_id','item_category','behavior_type'], \
    'last')[['user_id','item_category','behavior_type','uc_cumcount']]
uc_df_bcount = pd.get_dummies(uc_df_bcount['behavior_type']).join(uc_df_bcount[['user_id','item_category','uc_cumcount']])
uc_df_bcount.rename(columns = \
    {1:'behavior_type_1', 2:'behavior_type_2', 3:'behavior_type_3',4:'behavior_type_4'}, inplace=True)
uc_df_bcount['uc_b1count'] = uc_df_bcount['behavior_type_1'] * (uc_df_bcount['uc_cumcount']+1)
uc_df_bcount['uc_b2count'] = uc_df_bcount['behavior_type_2'] * (uc_df_bcount['uc_cumcount']+1)
uc_df_bcount['uc_b3count'] = uc_df_bcount['behavior_type_3'] * (uc_df_bcount['uc_cumcount']+1)
uc_df_bcount['uc_b4count'] = uc_df_bcount['behavior_type_4'] * (uc_df_bcount['uc_cumcount']+1)

uc_df_bcount_sum = uc_df_bcount.groupby(['user_id', 'item_category']) \
    .agg({'uc_b1count':np.sum, 'uc_b2count':np.sum, 'uc_b3count':np.sum, 'uc_b4count':np.sum})
uc_df_bcount_sum.reset_index(inplace=True)

uc_df_avg = uc_df_bcount_sum.groupby(['user_id']) \
    .agg({'uc_b1count':['mean','sum'], 'uc_b2count':['mean','sum'], 'uc_b3count':['mean','sum'], 'uc_b4count':['mean','sum']})
uc_df_avg.columns = ['_'.join(col).strip() for col in uc_df_avg.columns.values]
uc_df_avg.reset_index(inplace=True)

# uc_df
uc_df = pd.merge(uc_df_bcount_sum, uc_df_avg, how='left', on=['user_id'])
uc_df['uc_b1count_mean'] = (uc_df['uc_b1count']-uc_df['uc_b1count_mean'])/uc_df['uc_b1count_sum']
uc_df['uc_b2count_mean'] = (uc_df['uc_b2count']-uc_df['uc_b2count_mean'])/uc_df['uc_b2count_sum']
uc_df['uc_b3count_mean'] = (uc_df['uc_b3count']-uc_df['uc_b3count_mean'])/uc_df['uc_b3count_sum']
uc_df['uc_b4count_mean'] = (uc_df['uc_b4count']-uc_df['uc_b4count_mean'])/uc_df['uc_b4count_sum']
uc_df = uc_df[['user_id','item_category','uc_b1count_mean','uc_b2count_mean','uc_b3count_mean','uc_b4count_mean']]
print(uc_df.head())
pickle.dump(uc_df, open("./features/uc_df_{}.pyc".format(date_range),"wb"))

### u
# 每个用户对应的4个量及总量
user_df['u_cumcount'] = user_df.groupby(['user_id','behavior_type']).cumcount()
user_df_bcount = user_df.drop_duplicates(['user_id','behavior_type'], \
    'last')[['user_id','behavior_type','u_cumcount']]
user_df_bcount = pd.get_dummies(user_df_bcount['behavior_type']).join(user_df_bcount[['user_id','u_cumcount']])
user_df_bcount.rename(columns = \
    {1:'behavior_type_1', 2:'behavior_type_2', 3:'behavior_type_3',4:'behavior_type_4'}, inplace=True)
user_df_bcount['u_b1count'] = user_df_bcount['behavior_type_1'] * (user_df_bcount['u_cumcount']+1)
user_df_bcount['u_b2count'] = user_df_bcount['behavior_type_2'] * (user_df_bcount['u_cumcount']+1)
user_df_bcount['u_b3count'] = user_df_bcount['behavior_type_3'] * (user_df_bcount['u_cumcount']+1)
user_df_bcount['u_b4count'] = user_df_bcount['behavior_type_4'] * (user_df_bcount['u_cumcount']+1)
user_df_bcount = user_df_bcount.groupby('user_id') \
    .agg({'u_b1count':np.sum, 'u_b2count':np.sum, 'u_b3count':np.sum, 'u_b4count':np.sum})
user_df_bcount.reset_index(inplace=True)

user_df_bcount['b4_rate'] = user_df_bcount['u_b4count']/user_df_bcount['u_b1count']
user_df_bcount['b24_rate'] = user_df_bcount['u_b4count']/user_df_bcount['u_b2count']
user_df_bcount['b34_rate'] = user_df_bcount['u_b4count']/user_df_bcount['u_b3count']
u_df = user_df_bcount[['user_id','b4_rate','b24_rate','b34_rate']]

# 浏览过的购买比例及二次购买比例
user_df_viewed_item = ui_df_bcount.groupby('user_id')
total_count = user_df_viewed_item.apply(lambda x: x.count()[0])
user_df_bought_item = ui_df_bcount[ui_df_bcount['ui_b4count'] > 0].groupby('user_id')
bought_count = user_df_bought_item.apply(lambda x: x.count()[0])
item_b41_rate = (bought_count/total_count).fillna(0).to_frame()
item_b41_rate.rename(columns = {0:"item_b41_rate"}, inplace=True)

user_df_bought_twice_item = ui_df_bcount[ui_df_bcount['ui_b4count'] > 1].groupby('user_id')
bought_twice_count = user_df_bought_twice_item.apply(lambda x: x.count()[0])
item_b4_twice_rate = (bought_twice_count/bought_count).fillna(0).to_frame()
item_b4_twice_rate.rename(columns = {0:"item_b4_twice_rate"}, inplace=True)

u_df = pd.merge(u_df, item_b41_rate, how='left', on=['user_id'])
u_df = pd.merge(u_df, item_b4_twice_rate, how='left', on=['user_id'])

# 平均各类行为与购买的时间
user_item_first_purchase = user_df[user_df['behavior_type'] ==4 ].groupby(["user_id","item_id"])["time"].min().reset_index(name="user_item_first_purchase")
user_item_first_view = user_df[user_df['behavior_type'] ==1 ].groupby(["user_id","item_id"])["time"].min().reset_index(name="user_item_first_view")
user_item_gap_14 = pd.merge(user_item_first_purchase,user_item_first_view,how="inner",on=["user_id","item_id"])
user_item_gap_14["gap_14"] = user_item_gap_14["user_item_first_purchase"] - user_item_gap_14["user_item_first_view"]
user_item_gap_14["gap_14"] =user_item_gap_14["gap_14"].map(lambda x:x.days*24+x.seconds//3600)
user_item_gap_14["gap_14"] =user_item_gap_14["gap_14"].apply(lambda x:0.5 if x==0 else x)
user_item_gap_14["gap_14"] =user_item_gap_14["gap_14"].map(lambda x:7*24/x)
user_item_gap_14["gap_14"] =user_item_gap_14["gap_14"].apply(lambda x:1 if x<0 else x)
user_view_interval = user_item_gap_14.groupby("user_id")["gap_14"].mean().reset_index(name="user_view_interval")

user_item_first_save = user_df[user_df['behavior_type'] ==2 ].groupby(["user_id","item_id"])["time"].min().reset_index(name="user_item_first_save")
user_item_gap_24 = pd.merge(user_item_first_purchase,user_item_first_save,how="inner",on=["user_id","item_id"])
user_item_gap_24["gap_24"] = user_item_gap_24["user_item_first_purchase"] - user_item_gap_24["user_item_first_save"]
user_item_gap_24["gap_24"] = user_item_gap_24["gap_24"].map(lambda x:x.days*24+x.seconds//3600)
user_item_gap_24["gap_24"] =user_item_gap_24["gap_24"].apply(lambda x:0.5 if x==0 else x)
user_item_gap_24["gap_24"] =user_item_gap_24["gap_24"].map(lambda x:7*24/x)
user_item_gap_24["gap_24"] =user_item_gap_24["gap_24"].apply(lambda x:1 if x<0 else x)
user_save_interval = user_item_gap_24.groupby("user_id")["gap_24"].mean().reset_index(name="user_save_interval")

user_item_first_shoppingcart = user_df[user_df['behavior_type'] ==3 ].groupby(["user_id","item_id"])["time"].min().reset_index(name="user_item_first_shoppingcart")
user_item_gap_34 = pd.merge(user_item_first_purchase,user_item_first_shoppingcart,how="inner",on=["user_id","item_id"])
user_item_gap_34["gap_34"] = user_item_gap_34["user_item_first_purchase"] - user_item_gap_34["user_item_first_shoppingcart"]
user_item_gap_34["gap_34"] = user_item_gap_34["gap_34"].map(lambda x:x.days*24+x.seconds//3600)
user_item_gap_34["gap_34"] =user_item_gap_34["gap_34"].apply(lambda x:0.5 if x==0 else x)
user_item_gap_34["gap_34"] =user_item_gap_34["gap_34"].map(lambda x:7*24/x)
user_item_gap_34["gap_34"] =user_item_gap_34["gap_34"].apply(lambda x:1 if x<0 else x)
user_shoppingcart_interval = user_item_gap_34.groupby("user_id")["gap_34"].mean().reset_index(name="user_shoppingcart_interval")

u_df = pd.merge(u_df,user_view_interval,on="user_id", how="outer").fillna(0)
u_df = pd.merge(u_df,user_save_interval,on="user_id", how="outer").fillna(0)
u_df = pd.merge(u_df,user_shoppingcart_interval,on="user_id", how="outer").fillna(0)

pickle.dump(u_df, open("./features/u_df_{}.pyc".format(date_range),"wb"))
# print(user_df_bcount.count()[0])