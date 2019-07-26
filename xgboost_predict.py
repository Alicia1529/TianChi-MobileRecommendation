import pandas as pd 
import numpy as np

import xgboost as xgb
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import gc
import time


# 记录程序运行时间
start_time=time.time()

# 读入数据
data_name = "Dec13_Dec18"
path = "data/train_df_{}".format(data_name)
predict_df = pickle.load(open(path,"rb"))

model = xgb.Booster(model_file = "xgb.model")

feature_cols = [i for i in predict_df.columns if i not in ['user_id','item_id','item_category','label']]
dPredict = xgb.DMatrix(data=predict_df[feature_cols].values,feature_names=feature_cols)

#predicting
y_preds = (model.predict(dPredict) > 0.5).astype(int) 
print(len(y_preds))
predict_df["pred_label"] = y_preds 
predict_df[predict_df['pred_label'] == 1].to_csv("data/prediction_result_0.5_depth25.csv", 
                                                columns=['user_id','item_id'],
                                                index=False, header=True)



# watchlist = [(dPredict,'predict')]  # set valid set f1-score as the optimize objective
# evals_res = {}
# bst = None

# pks = pickle.dumps(bst)  # store the bst
# bst = pickle.loads(pks)


