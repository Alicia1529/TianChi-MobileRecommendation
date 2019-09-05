import pandas as pd 
import numpy as np

import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import gc
import time



# 记录程序运行时间
start_time=time.time()

# 读入数据
# val_df = pickle.load(open("data/train_df_Nov18_Nov25","rb"))
# train_df = pickle.load(open("data/train_df_Nov26_Dec3","rb"))
val_df = pickle.load(open("data/train_df_Nov26_Dec3","rb"))
train_df = pickle.load(open("data/train_df_Nov18_Nov25","rb"))



### definition of custom f1_score for training parameter feval
def f1_score(preds, dtrain):
    y_labels = dtrain.get_label()
    y_preds  = (preds > 0.5).astype(int) 
    return 'f1-score', metrics.f1_score(y_labels, y_preds)
#################################initial parameters#######################################


xg_params = {
        'eta':[0.1,0.01,0.001], #0.1,
        'max_depth':[6,10,20],  #6 #20(best)
        'min_child_weight':[1,2,3], #2
# 46 这个参数默认是1，是每个叶子里面h的和至少是多少，对正负样本不均衡时的0-1分类而言
# 47 假设h在0.01附近，min_child_weight为1意味着叶子节点中最少需要包含100个样本
# 48 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易overfitting。
        'gamma':[0.1,0.05] #0.1
    }

other_param = {
        'nthread':4,
        'seed':12,
        'n_estimators':10,
# 46 这个参数默认是1，是每个叶子里面h的和至少是多少，对正负样本不均衡时的0-1分类而言
# 47 假设h在0.01附近，min_child_weight为1意味着叶子节点中最少需要包含100个样本
# 48 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易overfitting。
        # 'subsample':1,
        # 'colsample_bytree':0.4,
        'lambda':1,
        # 'alpha':1,
        'scale_pos_weight':1,
        'booster':'gbtree',
        'objective':'binary:logistic',
        "feval":f1_score,
        'scale_pos_weight':1 #在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。
    }



# 1.1 tuning n_estimators(num_boost_round) with a relative high learning_rate(eta)
# xg_param['n_estimators'] = 10000
# 1.1.2 training and evaluating with fi-score curve (continues test)
feature_cols = [i for i in train_df.columns if i not in ['user_id','item_id','item_category','label']]
dtrain = xgb.DMatrix(data=train_df[feature_cols].values, label=train_df['label'].values, feature_names=feature_cols)
dval = xgb.DMatrix(data=val_df[feature_cols].values, label=val_df['label'].values, feature_names=feature_cols)

# del(train_df)
# gc.collect()

evals_res = {}
watchlist = [(dtrain,'train'), (dval, 'validate')]  # set valid set f1-score as the optimize objective
bst = None

gbm = xgb.XGBClassifier(**other_param)
opt_clf = GridSearchCV(estimator=gbm,param_grid=xg_params,cv=5)
opt_clf.fit(train_df[feature_cols].values, train_df['label'].values)
print('参数最佳取值:{0}'.format(opt_clf.best_params_))
print('最佳模型得分:{0}'.format(opt_clf.best_score_))
# bst = xgb.train(xg_param, dtrain, num_boost_round=xg_param['n_estimators'], early_stopping_rounds=400,
#                 evals=watchlist, feval=f1_score, maximize=True, evals_result=evals_res,
#                 xgb_model=bst, verbose_eval=True) 
                
# bst.save_model("xgb.model")
# bst.dump_model("dump.raw.txt", "featmap.txt")

# info visualization for judgment
# plt.figure(1)
# plt.plot(evals_res['train']['error'], label='train-loss')
# plt.plot(evals_res['validate']['error'], label='valid-loss')
# plt.plot(evals_res['train']['f1-score'], label='train-f1')
# plt.plot(evals_res['validate']['f1-score'], label='valid-f1')
# plt.xlabel('n_estimators')
# plt.ylabel('error_rate/f1-score')
# plt.title('error_rate/f1-score of training - XGB \n (eta=0.001 + default)')
# plt.legend()
# plt.grid(True, linewidth=0.5)
# plt.show()
