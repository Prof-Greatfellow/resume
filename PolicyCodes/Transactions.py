#%% 图3：根据张璇老师的建议，添加关于交易总量的信息
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
import gurobipy as gp
from gurobipy import GRB

#%% 根据张璇老师建议，在图3中添加关于交易总量的信息
# 交易总量怎么计算？我们还是需要拿回全国UHV和CBS交易总量的表格
mpl.rcParams['font.family'] = ['Arial'] #或者是Heiti TC
plt.rcParams['font.size']=18
plt.rcParams['axes.unicode_minus']=False
num_province=31
excel_name='../20240219整理文件/4.(测试2可相信）重新整合各省平衡表20240322.xlsx'
province_basicinfo=pd.read_excel(excel_name,sheet_name='2.ProvinceInfo',index_col=0,nrows=num_province,skiprows=[1,2])
province_property=province_basicinfo['Region'].to_dict() #东西南北中
province_list=list(province_basicinfo.index)
year_list=np.array([year for year in range(2025,2061)])
df_datasource=pd.read_excel(excel_name,sheet_name='1.SourceData',usecols='A:D',index_col=0,nrows=7410)
df_datasource['Power']/=10 #将单位从亿千瓦时改为TWh
df_transfer_byyear_list=[]
for year in year_list:
    df_transfer_byyear=pd.read_excel(excel_name,sheet_name=str(year),usecols='A:AH',
        skiprows=[1,2],nrows=num_province,index_col=0).fillna(0.0)
    df_transfer_byyear[['BalanceCheck','Load','Hydro','Nuclear','Solar','Wind',
       'Others','Gen-Load','RenewableOut','RenewableIn','Gen-Load-RenewOut+RenewIn',
       'HydroOut','HydroIn','CoalFire','CoalFireOut','CoalFireIn','RawCoalFire(SelfGen)',
       'NewCoalFire(SelfGen)','RawCoalPurchase','NewCoalPurchase','MaxSellCoal','MaxSellCoal→SellPower','MaxBuyCoal→BuyPower','SellCoal→SellPower',
       'BuyCoal→BuyPower','CapacityUHV','RemainUHVForCoals','RenewableOut(UHV)','RenewableIn(UHV)']]/=10 #将亿千瓦时改变为TWh，方便成文 
    df_transfer_byyear_list.append(df_transfer_byyear)

df_datasource_pivoted=df_datasource[df_datasource['Year']>=year_list[0]].reset_index().pivot_table(values='Power',index=['Province','Year'],columns='Type').fillna(0.0)

province_cn2en_dict=dict(zip(province_basicinfo['ProvinceCN'],province_basicinfo.index))
provincial_distance=pd.read_excel(excel_name,sheet_name='3.ProvincialDistance',usecols='A:AF',index_col=0)
curr_year=2040

provincial_distance=pd.read_excel(excel_name,sheet_name='3.ProvincialDistance',usecols='A:AF',index_col=0)
# 找到那个找最优解的程序
model=gp.Model("NetworkFlow")
model.Params.LogToConsole=0
# 考察以CBS方式运送煤炭的数量及最优化方法
df_transfer_curryear=df_transfer_byyear_list[curr_year-year_list[0]]
province_out=list(df_transfer_curryear[df_transfer_curryear['RenewableOut']-df_transfer_curryear['RenewableOut(UHV)']<0].index)
province_in=list(df_transfer_curryear[df_transfer_curryear['RenewableIn']-df_transfer_curryear['RenewableIn(UHV)']>0].index)
province_combination=[(i,j) for i in province_out for j in province_in]

province_flow=model.addVars(province_combination,lb=0.0,vtype=GRB.CONTINUOUS)
model.addConstrs((province_flow.sum(i,"*")==-(df_transfer_curryear.loc[i,'RenewableOut']-df_transfer_curryear.loc[i,'RenewableOut(UHV)']) for i in province_out),"FlowOut")
model.addConstrs((province_flow.sum("*",j)==(df_transfer_curryear.loc[j,'RenewableIn']-df_transfer_curryear.loc[j,'RenewableIn(UHV)']) for j in province_in),"FlowIn")
obj=sum(province_flow[(i,j)]*provincial_distance.loc[i,j] for (i,j) in province_combination)
model.setObjective(obj)
model.optimize()
df_result=pd.DataFrame(columns=['From','To','CBSAmount'])
for idx,(i,j) in enumerate(province_combination):
    df_result.loc[idx]=[i,j,province_flow[(i,j)].X]
df_result_positive=df_result[df_result['CBSAmount']>0].reset_index(drop=True)

province_out_UHV=list(df_transfer_curryear[df_transfer_curryear['RenewableOut(UHV)']<0].index)
province_in_UHV=list(df_transfer_curryear[df_transfer_curryear['RenewableIn(UHV)']>0].index)
province_combination_UHV=[(i,j) for i in province_out_UHV for j in province_in_UHV]
model_UHV=gp.Model("NetworkFlow")
model_UHV.Params.LogToConsole=0
province_flow_UHV=model_UHV.addVars(province_combination_UHV,lb=0.0,vtype=GRB.CONTINUOUS)
model_UHV.addConstrs((province_flow_UHV.sum(i,"*")==-df_transfer_curryear.loc[i,'RenewableOut(UHV)'] for i in province_out_UHV),"FlowOut")
model_UHV.addConstrs((province_flow_UHV.sum("*",j)==df_transfer_curryear.loc[j,'RenewableIn(UHV)'] for j in province_in_UHV),"FlowIn")
obj_UHV=sum(province_flow_UHV[(i,j)]*provincial_distance.loc[i,j] for (i,j) in province_combination_UHV)
model_UHV.setObjective(obj_UHV)
model_UHV.optimize()
df_result_UHV=pd.DataFrame(columns=['From','To','UHVAmount'])
for idx,(i,j) in enumerate(province_combination_UHV):
    df_result_UHV.loc[idx]=[i,j,province_flow_UHV[(i,j)].X]
df_result_UHV_positive=df_result_UHV[df_result_UHV['UHVAmount']>0].reset_index(drop=True)

#%% 用电公司首先要将钱交给售电公司
df_UHV_powersales