from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

import pandas as pd
import numpy
from itertools import combinations
import matplotlib.pyplot as plt

df = pd.read_csv('./data_to_transaction.csv',sep='delimiter', header=None,engine='python')
dataset= df #number of transactions

products_list = dataset.values.tolist()

for i in range(len(products_list)):
    string = products_list[i][0]
    products_list[i]=string.split(",")
#     products_list[i].pop()
    print(products_list[i])


# df = pd.read_csv('C:/Users/zoezo/Desktop/ntut_ms/CD/expe_zoe/transcation_data/data_to_transaction.csv', engine='python')
data = df.values.tolist()
# 轉換資料格式
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# # 設定最小支持度 (min_support) 來尋找頻繁項目集
# frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)

# # 生成關聯規則 (confidence, lift)
# rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# print(frequent_itemsets)
# print(rules)


# 設定 FP-Growth 來尋找頻繁項目集
frequent_itemsets_fp = fpgrowth(df_trans, min_support=0.01, use_colnames=True)

# 生成關聯規則
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.5)

print(frequent_itemsets_fp)
