# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


df_ = pd.read_excel('datasets/online_retail_II.xlsx', sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

_list = [a for a in df["StockCode"].unique() if str(a).isalpha()]
df[df["StockCode"].isin(_list)][["StockCode", "Description"]].groupby(["StockCode", "Description"]).agg({'StockCode': 'count'})


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe['Invoice'].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    #Numerik olmayan stock code'ları bul (POST, M, PADS, DOT)
    exc_list = [a for a in dataframe["StockCode"].unique() if str(a).isalpha()]
    #bu stock kodlu kayıtları df'den çıkar
    dataframe = dataframe[~dataframe["StockCode"].isin(exc_list)]
    return dataframe

df.shape
df = retail_data_prep(df)
df.describe().T

df.groupby(["Invoice", "StockCode"])[["StockCode"]].count()

#Q2
df_de = df[df["Country"] == "Germany"]

df_de.groupby(["Invoice", "Description"]).agg({'Quantity': 'sum'}).unstack()

df_de.groupby(["Invoice", "Description"]).agg({'Quantity': 'sum'}).unstack().fillna(0)

df_de.groupby(["Invoice", "Description"]). \
    agg({'Quantity': 'sum'}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0)


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

df_de_inv_pro = create_invoice_product_df(df_de, id=True)

def check_id(dataframe, stock_code):
    prod = dataframe[dataframe["StockCode"] == stock_code]["Description"].values[0]
    print(prod)


frequent_itemsets = apriori(df_de_inv_pro, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head()

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()

#Q3
item1 = 21987
item2 = 23235
item3 = 22747
items = [21987, 23235, 22747]

check_id(df_de, item1) #PACK OF 6 SKULL PAPER CUPS
check_id(df_de, item2) #STORAGE TIN VINTAGE LEAF
check_id(df_de, item3) #POPPY'S PLAYHOUSE BATHROOM

#Q4 & Q5

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]



arl_recommender(rules, item1, 1)
check_id(df_de, arl_recommender(rules, item1, 1)[0])

arl_recommender(rules, item2, 1)
check_id(df_de, arl_recommender(rules, item2, 1)[0])

arl_recommender(rules, item3, 1)
check_id(df_de, arl_recommender(rules, item3, 1)[0])
