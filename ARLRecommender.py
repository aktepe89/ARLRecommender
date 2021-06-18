#Import pandas and MLXtend
# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

#read dataset
df_ = pd.read_excel('datasets/online_retail_II.xlsx', sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

_list = [a for a in df["StockCode"].unique() if str(a).isalpha()]
df[df["StockCode"].isin(_list)][["StockCode", "Description"]].groupby(["StockCode", "Description"]).agg({'StockCode': 'count'})

#A function to find outliers of the variable passed to the argument and get them to the defined up and low limits
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

#A function to automatize replacing outliers
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#Data prep
#DropNAs, remove transactions with ID that contains "C" which implies the product is returned
#Remove rows where quantity and price is 0
#Replace outliers with threshold values
#Remove products without any numbers in the stock codes
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe['Invoice'].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    #Find stock codes that do not have any numbers in them (POST, M, PADS, DOT)
    exc_list = [a for a in dataframe["StockCode"].unique() if str(a).isalpha()]
    #Remove records with those stock codes
    dataframe = dataframe[~dataframe["StockCode"].isin(exc_list)]
    return dataframe

#Check DF
df.shape
df = retail_data_prep(df)
df.describe().T

df.groupby(["Invoice", "StockCode"])[["StockCode"]].count()

#Reduce DF to Germany only to work easier
df_de = df[df["Country"] == "Germany"]

#Create an invoice-product matrix. Fill NAs with zeros and change any numbers greater than 1 with 1s as we are only interested if the product is bought and not with quantity.
df_de.groupby(["Invoice", "Description"]). \
    agg({'Quantity': 'sum'}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0)

#The previous operation can also be turned into a function.
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

df_de_inv_pro = create_invoice_product_df(df_de, id=True)


#A function to get product description from stock code
def check_id(dataframe, stock_code):
    prod = dataframe[dataframe["StockCode"] == stock_code]["Description"].values[0]
    print(prod)

#Use apriori to find support values
frequent_itemsets = apriori(df_de_inv_pro, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head()

#Association rules to find other metrics such as lift, convict etc.
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()

#Get randomly selected 3 items
item1 = 21987
item2 = 23235
item3 = 22747

#Check their descriptions
check_id(df_de, item1) #PACK OF 6 SKULL PAPER CUPS
check_id(df_de, item2) #STORAGE TIN VINTAGE LEAF
check_id(df_de, item3) #POPPY'S PLAYHOUSE BATHROOM




#Create a recommender function
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]


#Pass the pre-selected items into the functions and get the recomended items
arl_recommender(rules, item1, 1)
check_id(df_de, arl_recommender(rules, item1, 1)[0])

arl_recommender(rules, item2, 1)
check_id(df_de, arl_recommender(rules, item2, 1)[0])

arl_recommender(rules, item3, 1)
check_id(df_de, arl_recommender(rules, item3, 1)[0])
