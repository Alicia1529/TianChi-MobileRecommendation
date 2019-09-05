df3 = pickle.load(open("data/train_df_Dec13_Dec18","rb"))
output = df3[df3["shoppingcart_notpurchase"]==1]
output = output[ ["user_id","item_id"] ]
output = output.drop_duplicates()
output.to_csv("data/shoppingcartNotBuy.csv",columns=["user_id","item_id"],index = False,header = True)