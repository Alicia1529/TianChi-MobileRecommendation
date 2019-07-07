import pandas as pd 
import numpy as np

import xgboost as xgb

from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import gc

data_train_item = "tianchi_fresh_comp_train_item.csv"
data_train_user = "tianchi_fresh_comp_train_user.csv"

#################################initial parameters#######################################
xg_param = {
        'nthread':4,
#         'seed':13,
        'eta':0.02,
#         'n_estimators':240,
#         'max_depth':6,
#         'min_child_weight':10,
#         'gamma':0.3,
#         'subsample':1,
#         'colsample_bytree':0.4,
#         'lambda':1,
#         'alpha':1,
#         'scale_pos_weight':1,
        'booster':'gbtree',
        'objective':'binary:logistic'
    }
# 1.1 tuning n_estimators(num_boost_round) with a relative high learning_rate(eta)
xg_param['n_estimators'] = 10000
# 1.1.2 training and evaluating with fi-score curve (continues test)
train_df, _ = data_set_construct_by_part(np_ratio = 70, sub_ratio = 1)
feature_cols = [i for i in train_df.columns if i not in ['user_id','item_id','item_category','label','class']]
dtrain = xgb.DMatrix(data=train_df[feature_cols].values, label=train_df['label'].values, feature_names=feature_cols)

_, valid_df = data_set_construct_by_part(np_ratio = 1200, sub_ratio = 0.4)
dvalid = xgb.DMatrix(data=valid_df[feature_cols].values, label=valid_df['label'].values, feature_names=feature_cols)   

watchlist = [(dtrain,'train'), (dvalid,'valid')]  # set valid set f1-score as the optimize objective

del(valid_df)
del(train_df)
gc.collect()

evals_res = {}
watchlist = [(dtrain,'train'), (dvalid,'valid')]  # set valid set f1-score as the optimize objective

bst = None
bst = xgb.train(xg_param, dtrain, num_boost_round=xg_param['n_estimators'], early_stopping_rounds=400,
                evals=watchlist, feval=f1_score, maximize=True, evals_result=evals_res,
                xgb_model=bst, verbose_eval=True) 

# pks = pickle.dumps(bst)  # store the bst
# bst = pickle.loads(pks)

# info visualization for judgment
plt.figure(1)
plt.plot(evals_res['train']['error'], label='train-loss')
plt.plot(evals_res['valid']['error'], label='valid-loss')
plt.plot(evals_res['train']['f1-score'], label='train-fi1')
plt.plot(evals_res['valid']['f1-score'], label='valid-f1')
plt.xlabel('n_estimators')
plt.ylabel('error_rate/f1-score')
plt.title('error_rate/f1-score of training - XGB \n (eta=0.1 + default)')
plt.legend()
plt.grid(True, linewidth=0.5)
plt.show()


