import pandas as pd
import numpy as np

ori_path = "../fresh_comp_offline/tianchi_fresh_comp_train_user.csv"

Nov18_Nov24_path = "../data/Nov18_Nov24.csv"
Nov25_path = "../data/Nov25.csv"
Nov26_Dec2_path = "../data/Nov26_Dec2.csv"
Dec3_path = "../data/Dec3.csv"
Dec13_Dec18_path = "../data/Dec13_Dec18.csv"

ori_df = pd.read_csv(ori_path).sort_values("time")
ori_df['date'] = ori_df['time'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H'))
ori_df = ori_df.set_index('date')
print(ori_df.head())

Nov18_Nov24 = ori_df['2014-11-18':'2014-11-24']
Nov25 = ori_df['2014-11-25']
Nov26_Dec2 = ori_df['2014-11-26':'2014-12-2']
Dec3 = ori_df['2014-12-3']
Dec13_Dec18 = ori_df['2014-12-13':'2014-12-18']

Nov18_Nov24.to_csv(Nov18_Nov24_path, \
    columns=['user_id','item_id','behavior_type','user_geohash','item_category','time'])
Nov25.to_csv(Nov25_path, \
    columns=['user_id','item_id','behavior_type','user_geohash','item_category','time'])
Nov26_Dec2.to_csv(Nov26_Dec2_path, \
    columns=['user_id','item_id','behavior_type','user_geohash','item_category','time'])
Dec3.to_csv(Dec3_path, \
    columns=['user_id','item_id','behavior_type','user_geohash','item_category','time'])
Dec13_Dec18.to_csv(Dec13_Dec18_path, \
    columns=['user_id','item_id','behavior_type','user_geohash','item_category','time'])   
