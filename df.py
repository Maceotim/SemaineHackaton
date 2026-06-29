import pandas as pd

data = pd.ExcelFile("data_CRIR_anon.xlsx")

data_P = pd.read_excel(data, 'P')
data_S = pd.read_excel(data, 'S')
data_EF = pd.read_excel(data, 'EF')
data_T = pd.read_excel(data, 'T')

data_EF_clean = data_EF.drop(data_EF[data_EF['P6_reel']==1.00].index)
data_EF_clean.drop(0, inplace=True)

data_P.drop(0, inplace=True)
data_EF_clean['round_P6']= data_EF_clean['P6_reel'].round(0)
df= pd.merge(data_EF_clean, data_P, how='left', left_on=['essai','round_P6'] , right_on=['essai','P6_target'])
df.head(25)