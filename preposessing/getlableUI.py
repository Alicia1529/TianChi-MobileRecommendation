import pandas as pd
import numpy as np
from datetime import datetime
import pickle

filename = "Nov18_Nov24"

user_file = "data/{}.csv".format(filename)

user_history = pd.read_csv(user_history)

user_purchase = user_history[user_history['behavior_type'] ==4][["user_id","item_id"]]
user_purchase["label"] = 1

user_item = user_history.drop_duplicates(["user_id","item_id"])[["user_id","item_id"]]

labeled_data = pd.merge(user_item,user_purchase,on=["user_id","item_id"], how="outer").fillna(0)

outputname = "featurs/{}_lable.pyc".format(filename)

pickle.dump(labeled_data, open(outputname,"wb"))





