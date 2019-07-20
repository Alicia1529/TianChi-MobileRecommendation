import pandas as pd
import numpy as np

ori_path = "../fresh_comp_offline/tianchi_fresh_comp_train_user.csv"

Nov18_Nov24_path = "data/Nov18_Nov24.csv"
Nov25_path = "data/Nov25.csv"
Nov26_Dec2_path = "data/Nov26_Dec2.csv"
Dec3_path = "data/Dec3.csv"
Dec13_Dec18_path = "data/Dec13_Dec18.csv"

ori_df = pd.read_csv(ori_path).sort_values("time")
ori_df['date'] = ori_df['time'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H'))
print(ori_df.head())
