
#%% 0.1 注脚
# 目前我们对电价的理解是有误的。各省赚得的钱是根据对方省份的电价来定的，不是根据自己省份的电价来定的
# 其实只有SelfUse_Rate_RenewableGen是自己省份的电价
 
#%% 0.2 导入基本的包
#（参考6.成本分析-光伏储能公司20240405。结果在小数点后三位左右有不一致，但可以接受。）
# 由于原表格有各种遗漏和疏忽，所以我们重新放进
#???难道final_capacity不能用来进行二次回收/售卖吗???
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB,max_
import sys
import geopandas as gpd
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.patches import Arrow
num_province=31
province_basicinfo=pd.read_excel('../数据整理/各省基本信息（英文版）.xlsx',index_col=1,nrows=num_province)
province_list=list(province_basicinfo.index)
plt.rcParams['font.size']=24
plt.rcParams['font.family']='Arial'

Exchange_rate=7 # 汇率
#%% 1.1 首先设定一些基本参数和函数
year_list=np.arange(2025,2061) # ref列表存储的是2023年开始的数据
year_base=year_list[0]-1
#SelfUse_rate_PV=0.25 #假设自用比例为25%（可供自身工厂使用，也可给d附近园区使用），外发比例为75%

# 峰值/谷值单价：不区分年份，目前也不区分省份；但后续要区分
Power_price_peak=province_basicinfo['PowerPricePeak']
#Power_price_peak.loc['Zhejiang']+=0.1
#Power_price_peak.loc['Guangdong']+=0.2
Power_price_norm=province_basicinfo['PowerPriceNorm']
Power_price_valley=province_basicinfo['PowerPriceValley']

#!!!修改Power_price_expansion的数值，使其成为从1.01开始指数增长的数组
Power_price_expansion_coef=np.concatenate((np.ones(len(year_list)//4)*1.02,np.ones(len(year_list)-len(year_list)//4)*0.99))
Power_price_expansion=np.ones_like(Power_price_expansion_coef)
for idx_year in range(1,len(year_list)):
    Power_price_expansion[idx_year]=Power_price_expansion[idx_year-1]*Power_price_expansion_coef[idx_year]

#!!!修改Power_price_peak_matrix等数值，使其已改为真实的各省值，而非全国统一的值
Power_price_peak_matrix=np.outer(Power_price_peak,Power_price_expansion)/Power_price_expansion[0]
Power_price_norm_matrix=np.outer(Power_price_norm,Power_price_expansion)/Power_price_expansion[0]
Power_price_valley_matrix=np.outer(Power_price_valley,Power_price_expansion)/Power_price_expansion[0]
df_PowerPricePeak=pd.DataFrame(Power_price_peak_matrix,index=province_list,columns=year_list)
Discount_rate=0.0 #??? 贴现率，按现在来看还不涉及到这一层，但一般来说会是6%
SelfUse_electricity_price=(Power_price_peak_matrix+Power_price_norm_matrix)/2

#!!! 假设输出电价和SelfUse_electricity_price相同，其实是(peak+norm)/2。最简单的方法是：
TransOut_electricity_price=SelfUse_electricity_price

# 这里重新假设，由于主要建立的是东部分布式光伏和西部集中式光伏，所以地价和工资是不影响各省的光伏建造成本的
HousingPriceRef=province_basicinfo['AverageHousingPrice']

HousingPriceRef=np.log(HousingPriceRef+1)
HousingRatio=0.3
WageRef=province_basicinfo['AverageWage']

WageRef=np.log(WageRef+1)
WageRatio=0.2
#%% 1.2 关于光伏的参数设定
CoDeployment_Rate=0.15

YR_depreciation_PV=10 # 发电设备（如光伏）的折旧年限
RES_value_PV=0.05 # 残值率
RT_depreciation_PV=(1-RES_value_PV)/YR_depreciation_PV #年折旧率
VAT_TAX=0.13
PV_Rate_Hayward=1.85 #单位：MW
#!!! 修改了借贷成本的偿还时间
YR_Loan_PV_list=np.array([max(4-idx_year//2,0) for idx_year in range(len(year_list))])

# 装机成本（增值税后）
#!!! 将PV_install_price的变迁从手动设置变成自动变化
#PV_install_price_max,PV_install_price_min=6000000,4260000
# 2022 年，我国地面光伏系统的初始全投资成本为4.13元/W左右,预计到2025年开始是3.1元/W左右
PV_install_price_max=3.1*1e6*PV_Rate_Hayward # 根据老妈整理，出处可能是https://www.stcn.com/article/detail/1139068.html
PV_install_price_min=PV_install_price_max*0.8
YR_usage_PV=25
PROVINCE_REF='Jiangsu'
#!!! 光伏年发电量区分不同的省份
Solar_generation_byprovince=[2550045.0/province_basicinfo.loc[PROVINCE_REF,'SolarIrradiance']*province_basicinfo.loc[province,'SolarIrradiance'] for province in province_list]

#下面也试一下风电的情况(暂时也借用一下光伏的变量名称)
# PV_rate_temp=2380 #238万千瓦=2380MW
# PV_install_price_max=155e8/PV_rate_temp*PV_Rate_Hayward # 马士基在内蒙古通辽市的一个风电项目，238MW 177亿元,10年运行期，需要
# PV_install_price_min=PV_install_price_max*0.8
# YR_usage_PV=20
# PROVINCE_REF='Jiangsu'
# Solar_generation_byprovince=[2500000*PV_Rate_Hayward for province in province_list]
#Solar_generation_byprovince=[2500000*PV_Rate_Hayward/province_basicinfo.loc[PROVINCE_REF,'WindDensity']*province_basicinfo.loc[province,'WindDensity'] for province in province_list] # 按年发电量2500小时进行计算

PV_install_price_list=PV_install_price_min+(PV_install_price_max-PV_install_price_min)*np.exp(-np.arange(len(year_list))*0.15)
df_PV_install_price=pd.DataFrame(index=province_list,columns=year_list)
for province in province_list:
    df_PV_install_price.loc[province]=PV_install_price_list*((1-HousingRatio-WageRatio)+HousingRatio*HousingPriceRef.loc[province]/HousingPriceRef.loc[PROVINCE_REF]+WageRatio*WageRef.loc[province]/WageRef.loc[PROVINCE_REF])
    #df_PV_install_price.loc[province]=PV_install_price_list

# 年运维OM和管理Admin成本百分比
#!!! 将OM_rate_list_PV变为统一的0.04
OM_rate_list_PV=np.ones_like(year_list,dtype=float)*0.05

# 年银行还贷百分比
Loan_rate_list_PV=np.ones_like(year_list,dtype=float)*0.03

# 政府补贴(元/kWh)
# 我们做一个新的假设：政府补贴增加至0.3元/kWh并每年减少0.05元/kWh，但只在初期补贴给那些需要运输CBS的企业
#Subsidy_for_CBS_byyear=np.array([max(0.3-0.05*idx_year,0.0) for idx_year in range(len(year_list))])
YR_Subsidy=6
Subsidy_for_CBS_byyear=np.zeros_like(year_list,dtype=float)
Subsidy_for_CBS_byyear[:YR_Subsidy//2]=0.1
Subsidy_for_CBS_byyear[YR_Subsidy//2:YR_Subsidy]=0.05

#!!! 假定Service Provider要对发电企业抽成！这点在之前的代码里竟然都没有提到！
#ServiceProvision_ChargeRate_byyear=0.2+0.3*np.exp((year_list[0]-year_list)/20)
#%% 1.3 关于储能的参数设定
# 储能初始容量（单位：MWh）
RES_value_CBS=0.8 # 残值率
YR_usage_CBS=5

Railway_largeorder_discount=0.9
Railway_price_yearly_increase=0.01
Railway_unit_cost=2.754*np.power(1+Railway_price_yearly_increase,np.arange(len(year_list)))*Railway_largeorder_discount #铁路40尺集装箱单位公里货运价格，单位：元/km
Railway_base_cost=680 #铁路40尺集装箱基准货运价格，单位：元

# 从官网pdf附录5获取https://www.cma.gov.cn/zfxxgk/gknr/qxbg/202402/t20240222_6082082.html

CBS_RoundTrip_Eff=0.98  # 储能循环效率
Railway_Return_Discount=0.5  # 返程打折比例
Railway_VAT_TAX=0.09    # 铁路运输增值税
AVG_Capacity_Coeff=(1+RES_value_CBS)/2 #使用过程中的平均容量(啥子玩意儿？)
CBS_SecondLife_PriceRate=0.75  # 储能二次利用的回收成本折扣

# CBS电柜的报价随年份的变化
#!!! Initial_Capacity_CBS_list采用了新型的连续值，且容量也乘以了2
# Initial_Capacity_CBS_list=np.floor(np.linspace(13.0,26.0,len(year_list)))
Initial_Capacity_CBS_list=np.floor(np.linspace(14.0,28.0,len(year_list)))
Rental_times_CBS_list=100*np.floor(np.linspace(60,70,len(year_list)))

# 休整过后的储能柜容量变化，第一个维度是。注意这个矩阵并不是方阵，因为2060年储能柜理论上可以用到2065年
Capacity_matrix_CBS=np.full((len(year_list),len(year_list)+YR_usage_CBS-1),np.nan)
for idx_year,year in enumerate(year_list):
    max_idx_CBS_ServiceRange=idx_year+YR_usage_CBS
    #max_idx_CBS_ServiceRange=min(idx_year+YR_usage_CBS,len(year_list))
    Capacity_matrix_CBS[idx_year,idx_year:max_idx_CBS_ServiceRange]=np.linspace(Initial_Capacity_CBS_list[idx_year],Initial_Capacity_CBS_list[idx_year]*RES_value_CBS,max_idx_CBS_ServiceRange-idx_year)

#!!! 由于CBS的初始容量变为了原来的两倍，所以price自然也要变为原来的两倍。且价格也变为连续下降的了

# CBS_install_price_max,CBS_install_price_min=10.8e6,8.4e6
# 参考《4h储能系统最新报价，低至0.511元/Wh》
CBS_install_price_max=0.5*Initial_Capacity_CBS_list[0]*1e6*1.13
CBS_install_price_min=CBS_install_price_max*0.8
CBS_install_price_list=CBS_install_price_min+(CBS_install_price_max-CBS_install_price_min)*np.exp(-np.arange(len(year_list))*0.1)

OM_rate_list_CBS=np.ones_like(year_list,dtype=float)*0.05
Loan_rate_list_CBS=np.ones_like(year_list,dtype=float)*0.03

Profit_rate_CBS=0.15
INCOME_TAX_CBS=0.1

# 特高压容量价格（每公里每千瓦时多少人民币）
# 按照发改委相关文件（如https://www.ndrc.gov.cn/xxgk/zcfb/tz/202204/t20220412_1321938.html），默认是0.000078元/度/公里
# !! 但也可以是自行设定特高压每年的度电成本！（很麻烦，也不太好实现，不太推荐）
UHV_LCOE=0.0001 #元/千瓦时/公里

##%% 再次导入之前的关于省份之间能量传输的表格
excel_name='../20240219整理文件/4.(测试2可相信）重新整合各省平衡表20240322.xlsx'
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

provincial_distance=pd.read_excel(excel_name,sheet_name='3.ProvincialDistance',usecols='A:AF',index_col=0)

# 这里重新写出SelfUse_Rate,Out_Rate_UHV以及Out_Rate_CBS（均以矩阵的形式体现）
SelfUse_Rate_RenewableGen=pd.DataFrame(index=province_list,columns=year_list)
Out_Rate_UHV_RenewableGen=pd.DataFrame(index=province_list,columns=year_list)
Out_Rate_CBS_RenewableGen=pd.DataFrame(index=province_list,columns=year_list)

for idx_year,year in enumerate(year_list):
    df_transfer_byyear=df_transfer_byyear_list[idx_year].loc[province_list]
    renewable_gen=df_transfer_byyear['Wind']+df_transfer_byyear['Solar']
    renewable_out_cbs=(df_transfer_byyear['RenewableOut(UHV)']-df_transfer_byyear['RenewableOut'])/renewable_gen
    renewable_out_uhv=-df_transfer_byyear['RenewableOut(UHV)']/renewable_gen
    selfuse_rate=1-renewable_out_cbs-renewable_out_uhv
    SelfUse_Rate_RenewableGen[year]=selfuse_rate
    Out_Rate_UHV_RenewableGen[year]=renewable_out_uhv
    Out_Rate_CBS_RenewableGen[year]=renewable_out_cbs

# 售电公司根据Revenue抽取1%的收入
ServiceFee_Rate=0.01

df_renewable_generation=pd.DataFrame(dtype=float,index=province_list,columns=year_list)
df_renewable_generation_CBSout=pd.DataFrame(dtype=float,index=province_list,columns=year_list)
province_renewable_transferin_list=[]
#province_renewable_transferout_list=[]
for idx_year,year in enumerate(year_list):
    df_transfer_curryear=df_transfer_byyear_list[idx_year]
    df_renewable_generation[year]=df_transfer_curryear.loc[province_list,'Solar']+df_transfer_curryear.loc[province_list,'Wind']
    df_renewable_generation_CBSout[year]=df_transfer_curryear.loc[province_list,'RenewableOut(UHV)']-df_transfer_curryear.loc[province_list,'RenewableOut']
    province_renewable_transferin_list.append(df_transfer_curryear.index[df_transfer_curryear['RenewableIn']>0].tolist())
    #province_renewable_transferout_list.append(df_transfer_curryear.index[df_transfer_curryear['RenewableOut']<0].tolist())

#%% 2.1 [铁路与特高压]最优路线规划和度电成本
# 首先还是用之前的gurobi建模优化方法，来得知每年的最优运输解
    
# 那么现在就是不同光伏公司的铁路成本。#
# 如果是非跨省传输，那么无需增加铁路运输成本。
# 如果是跨省传输，那么铁路运输成本还需增加
#!!! 修改了铁路运输的成本，将储能容量变成了动态的

df_CBStransfer_result_list=[]
province_transfer_profile_list_from=[]
province_transfer_profile_list_to=[]

for idx_year,year in enumerate(year_list):
    model=gp.Model("NetworkFlow(CBS)")
    model.Params.LogToConsole=0
    # 考察以CBS方式运送煤炭的数量及最优化方法
    df_transfer_curryear=df_transfer_byyear_list[idx_year]
    province_out=list(df_transfer_curryear[df_transfer_curryear['RenewableOut']-df_transfer_curryear['RenewableOut(UHV)']<0].index)
    province_in=list(df_transfer_curryear[df_transfer_curryear['RenewableIn']-df_transfer_curryear['RenewableIn(UHV)']>0].index)
    province_combination=[(i,j) for i in province_out for j in province_in]
    province_flow=model.addVars(province_combination,lb=0.0,vtype=GRB.CONTINUOUS)
    model.addConstrs((province_flow.sum(i,"*")==-(df_transfer_curryear.loc[i,'RenewableOut']-df_transfer_curryear.loc[i,'RenewableOut(UHV)']) for i in province_out),"FlowOut")
    model.addConstrs((province_flow.sum("*",j)==(df_transfer_curryear.loc[j,'RenewableIn']-df_transfer_curryear.loc[j,'RenewableIn(UHV)']) for j in province_in),"FlowIn")
    obj=sum(province_flow[(i,j)]*provincial_distance.loc[i,j] for (i,j) in province_combination)
    model.setObjective(obj)
    model.optimize()
    df_result=pd.DataFrame(columns=['From','To','CBSPower','Distance'])
    for idx,(i,j) in enumerate(province_combination):
        df_result.loc[idx]=[i,j,province_flow[(i,j)].X,provincial_distance.loc[i,j]]
        df_result_positive=df_result[df_result['CBSPower']>0].reset_index(drop=True)
    df_result_positive['CBSPowerXDistance']=df_result_positive['CBSPower']*df_result_positive['Distance']
    df_CBStransfer_result_list.append(df_result_positive)
    province_transfer_profile_list_from.append(df_result_positive.groupby('From').agg({'CBSPower':sum,'Distance':sum,'CBSPowerXDistance':sum}))
    province_transfer_profile_list_to.append(df_result_positive.groupby('To').agg({'CBSPower':sum,'Distance':sum,'CBSPowerXDistance':sum}))  


# 这里是否考虑一下，在计算df_LCOS_Railway的时候，考虑一下国家对西部省份的协调呢？
LCOS_Railway_matrix_dict={province:np.full((len(year_list),len(year_list)),np.nan) for province in province_list}
for idx_year,year in enumerate(year_list): #这里的year是真实年份
    province_transfer_profile_from=province_transfer_profile_list_from[idx_year]
    for province in province_list:
        idx_usable_CBS_version_list=np.arange(max(0,idx_year-YR_usage_CBS+1),idx_year+1)
        if province in province_transfer_profile_from.index:
            weighted_distance=province_transfer_profile_from.loc[province,'CBSPowerXDistance']/province_transfer_profile_from.loc[province,'CBSPower'] # 计算加权里程
            weighted_price=weighted_distance*Railway_unit_cost[idx_year]+Railway_base_cost
            for idx_version in idx_usable_CBS_version_list:
                # 储能容量的公式和价格有所改变
                LCOS_Railway_matrix_dict[province][idx_version,idx_year]=weighted_price/(Capacity_matrix_CBS[idx_version,idx_year]*1000*CBS_RoundTrip_Eff*AVG_Capacity_Coeff)*(1+Railway_Return_Discount)/(1+Railway_VAT_TAX)
        else:
            for idx_version in idx_usable_CBS_version_list:
                LCOS_Railway_matrix_dict[province][idx_version,idx_year]=0.0

df_LCOS_Railway_aggregated=pd.DataFrame(index=province_list,columns=year_list,dtype=float)
for province in province_list:
    df_LCOS_Railway_aggregated.loc[province]=np.nanmean(LCOS_Railway_matrix_dict[province],axis=0)

# 我们不严格按照铁路严格每年利润调整的加权平均值为0，而是对铁路
df_LCOS_Railway_adjustment=pd.DataFrame(0.0,index=province_list,columns=year_list) #绝对的调整值
weights_Railway_yearAvg=df_renewable_generation*Out_Rate_CBS_RenewableGen
df_LCOS_Railway_weightedAvg=pd.Series(0.0,index=year_list)
for year in year_list:
    df_LCOS_Railway_weightedAvg.loc[year]=np.average(df_LCOS_Railway_aggregated[year],weights=weights_Railway_yearAvg[year])
LCOS_Railway_normalized=df_LCOS_Railway_aggregated-df_LCOS_Railway_weightedAvg
df_LCOS_Railway_adjustment=-LCOS_Railway_normalized*0.5 # 设置5成的铁路运输价格调整
df_LCOS_Railway_aggregated+=df_LCOS_Railway_adjustment

# UHV大致应该怎么建设呢？也可以做一个类似的优化模型
df_UHV_transfer_result_list=[]
province_transfer_profile_list_from_UHV=[]
province_transfer_profile_list_to_UHV=[]
for idx_year,year in enumerate(year_list):
    model=gp.Model("NetworkFlow(UHV)")
    model.Params.LogToConsole=0
    # 考察以UHV方式运输电能的最优方法（距离x电量最少）
    df_transfer_curryear=df_transfer_byyear_list[idx_year]
    province_out=list(df_transfer_curryear[df_transfer_curryear['RenewableOut(UHV)']<0].index)
    province_in=list(df_transfer_curryear[df_transfer_curryear['RenewableIn(UHV)']>0].index)
    province_combination=[(i,j) for i in province_out for j in province_in]
    province_flow=model.addVars(province_combination,lb=0.0,vtype=GRB.CONTINUOUS)
    model.addConstrs((province_flow.sum(i,"*")==-df_transfer_curryear.loc[i,'RenewableOut(UHV)'] for i in province_out),"FlowOut")
    model.addConstrs((province_flow.sum("*",j)==df_transfer_curryear.loc[j,'RenewableIn(UHV)'] for j in province_in),"FlowIn")
    obj=sum(province_flow[(i,j)]*provincial_distance.loc[i,j] for (i,j) in province_combination)
    model.setObjective(obj)
    model.optimize()
    df_result=pd.DataFrame(columns=['From','To','UHVPower','Distance'])
    for idx,(i,j) in enumerate(province_combination):
        df_result.loc[idx]=[i,j,province_flow[(i,j)].X,provincial_distance.loc[i,j]]
        df_result_positive=df_result[df_result['UHVPower']>0].reset_index(drop=True)
    df_result_positive['UHVPowerXDistance']=df_result_positive['UHVPower']*df_result_positive['Distance']
    df_UHV_transfer_result_list.append(df_result_positive)
    province_transfer_profile_list_from_UHV.append(df_result_positive.groupby('From').agg({'UHVPower':sum,'Distance':sum,'UHVPowerXDistance':sum}))
    province_transfer_profile_list_to_UHV.append(df_result_positive.groupby('To').agg({'UHVPower':sum,'Distance':sum,'UHVPowerXDistance':sum}))  
    
# 如果这个地方不输出新能源，那么就是0.0
df_UHV_LCOE_aggregated=pd.DataFrame(0.0,index=province_list,columns=year_list)
for idx_year,year in enumerate(year_list):
    for province_from in province_transfer_profile_list_from_UHV[idx_year].index:
        df_UHV_LCOE_aggregated.loc[province_from,year]=province_transfer_profile_list_from_UHV[idx_year].loc[province_from,'UHVPowerXDistance']/province_transfer_profile_list_from_UHV[idx_year].loc[province_from,'UHVPower']*UHV_LCOE

#%% 2.2 [补贴]计算
# 根据CBS Transfer来决定每个省份具体需要多少的Subsidy
# 目前Subsidy为0
df_Subsidy_PV=pd.DataFrame(0.0,index=province_list,columns=year_list)
for idx_year,year in enumerate(year_list):
    province_list_from_curryear=province_transfer_profile_list_from[idx_year].index
    ratio_list=province_transfer_profile_list_from[idx_year]['CBSPowerXDistance']/province_transfer_profile_list_from[idx_year]['CBSPower']*Out_Rate_CBS_RenewableGen.loc[province_list_from_curryear,year]
    base=Subsidy_for_CBS_byyear[idx_year]*df_renewable_generation[year].sum()/(df_renewable_generation.loc[province_list_from_curryear,year]*ratio_list).sum()
    for province_from in province_list_from_curryear:
        df_Subsidy_PV.loc[province_from,year]=base*ratio_list.loc[province_from]
#df_Subsidy_PV=pd.DataFrame(0.0,index=province_list,columns=year_list)

#%% 2.3 [发电公司]收入和成本
# 首先做一个额外的步骤，查看TransOut_CBS和TransOut_UHV对各省的加权度电收入是多少（如果该省不输出则为0）
df_CBS_TransOut_AvgRev=pd.DataFrame(0.0,index=province_list,columns=year_list)
df_UHV_TransOut_AvgRev=pd.DataFrame(0.0,index=province_list,columns=year_list)

for idx_year,year in enumerate(year_list): #实际年份
    for province_from in province_transfer_profile_list_from[idx_year].index:
        df_filter_provincefrom=df_CBStransfer_result_list[idx_year][df_CBStransfer_result_list[idx_year]['From']==province_from].copy().reset_index(drop=True)
        df_filter_provincefrom['Price']=0.0
        for idx_province_to in range(df_filter_provincefrom.shape[0]):
            province_to=df_filter_provincefrom.loc[idx_province_to,'To']
            df_filter_provincefrom.loc[idx_province_to,'Price']=TransOut_electricity_price[province_list.index(province_to),idx_year]
        AvgPrice=np.average(df_filter_provincefrom['Price'],weights=df_filter_provincefrom['CBSPower'])
        df_CBS_TransOut_AvgRev.loc[province_from,year]=AvgPrice
    for province_from in province_transfer_profile_list_from_UHV[idx_year].index:
        df_filter_provincefrom=df_UHV_transfer_result_list[idx_year][df_UHV_transfer_result_list[idx_year]['From']==province_from].copy().reset_index(drop=True)
        df_filter_provincefrom['Price']=0.0
        for idx_province_to in range(df_filter_provincefrom.shape[0]):
            province_to=df_filter_provincefrom.loc[idx_province_to,'To']
            df_filter_provincefrom.loc[idx_province_to,'Price']=TransOut_electricity_price[province_list.index(province_to),idx_year]
        AvgPrice=np.average(df_filter_provincefrom['Price'],weights=df_filter_provincefrom['UHVPower'])
        df_UHV_TransOut_AvgRev.loc[province_from,year]=AvgPrice

# 然后算发电公司的发电收入和成本（暂时不考虑铁路运输、储能柜租赁和服务费）
LCOE_matrix_byprovince_dict=dict()
LROE_matrix_byprovince_dict=dict()
LROE_TransOut_matrix_byprovince_dict=dict()
for idx_province_from,province_from in enumerate(province_list):
    df_curryear_PV_list=[]
    for idx_year,year in enumerate(year_list): #该年份生产的光伏设备
        year_list_operation_PV=np.arange(year,min(year+YR_usage_PV,year_list[-1]+1))
        PV_install_price=np.array(df_PV_install_price.loc[province_from,year])
        Depreciation_byyear_PV=np.array([0 if i-year>=YR_depreciation_PV else PV_install_price/(1+VAT_TAX)*RT_depreciation_PV for i in year_list_operation_PV])
        OM_byyear_PV=PV_install_price/(1+VAT_TAX)*OM_rate_list_PV[year_list_operation_PV[0]-year_base-1:year_list_operation_PV[-1]-year_base] 
        #Admin_byyear_PV=PV_install_price/(1+VAT_TAX)*Admin_rate_list_PV[year_list_operation_PV[0]-year_base-1:year_list_operation_PV[-1]-year_base] 
        Loan_byyear_PV=np.array([0 if i-year>=YR_Loan_PV_list[idx_year] else PV_install_price*Loan_rate_list_PV[i-year_base-1] for i in year_list_operation_PV])
        Total_power_byyear=np.ones((len(year_list_operation_PV),),dtype=float)*Solar_generation_byprovince[idx_province_from]
        SelfUse_revenue_PV=Total_power_byyear*np.array(SelfUse_Rate_RenewableGen.loc[province_from,year_list_operation_PV])*np.array([SelfUse_electricity_price[idx_province_from,i-year_base-1] for i in year_list_operation_PV])/(1+VAT_TAX)
        # 为了计算TransOut_revenue_PV_CBS和TransOut_revenue_PV_UHV,我们需要回到最开始的Gurobi优化结果，来看表格里涉及了多少该省份的数据
        TransOut_revenue_PV_UHV=Total_power_byyear*np.array(Out_Rate_UHV_RenewableGen.loc[province_from,year_list_operation_PV])*\
         np.array(df_UHV_TransOut_AvgRev.loc[province_from,year_list_operation_PV])
        TransOut_revenue_PV_CBS=Total_power_byyear*np.array(Out_Rate_CBS_RenewableGen.loc[province_from,year_list_operation_PV])*\
         np.array(df_CBS_TransOut_AvgRev.loc[province_from,year_list_operation_PV])
        #TransOut_revenue_PV_UHV=Total_power_byyear*np.array(Out_Rate_UHV_RenewableGen.loc[province_from,year_list_operation_PV])*np.array([TransOut_electricity_price[idx_province_from,i-year_base-1] for i in year_list_operation_PV])/(1+VAT_TAX)
        #TransOut_revenue_PV_CBS=Total_power_byyear*np.array(Out_Rate_CBS_RenewableGen.loc[province_from,year_list_operation_PV])*np.array([TransOut_electricity_price[idx_province_from,i-year_base-1] for i in year_list_operation_PV])/(1+VAT_TAX) 
        Subsidy_byyear_PV=np.array(df_Subsidy_PV.loc[province_from,list(np.arange(year,year_list_operation_PV[-1]+1))])
        #???这里先设定ProfitTransfer为0
        ProfitTransfer_byyear_PV=np.zeros_like(year_list_operation_PV)
        #ProfitTransfer_byyear_PV=np.array(df_ProfitTransfer.loc[province_from,list(np.arange(year,year_list_operation_PV[-1]+1))])
        #PV侧的基础度电成本包括了折旧、运维、管理成本和银行贷款
        LCOE_PV=(Depreciation_byyear_PV+OM_byyear_PV+Loan_byyear_PV)/Total_power_byyear
         # levelized revenue of electricity    
        LROE_PV=(SelfUse_revenue_PV+TransOut_revenue_PV_UHV+TransOut_revenue_PV_CBS)/Total_power_byyear+Subsidy_byyear_PV+ProfitTransfer_byyear_PV
        # 因为售电公司只对输出的电能抽成，所以这里设定一个LROE_PV_TransOut，删掉了TransOut的部分
        LROE_TransOut_PV=(TransOut_revenue_PV_UHV+TransOut_revenue_PV_CBS)/Total_power_byyear+ProfitTransfer_byyear_PV
        df_curryear_PV=pd.DataFrame(index=year_list_operation_PV,columns=['Subsidy-PV','ProfitTransfer-PV','Depreciation-PV','OM-PV','TotalPower','SelfUseRevenue-PV','TransOutRevenue-PV','LCOE-PV','LROE-PV'])
        df_curryear_PV['Subsidy-PV']=Subsidy_byyear_PV
        df_curryear_PV['ProfitTransfer-PV']=ProfitTransfer_byyear_PV
        df_curryear_PV['Depreciation-PV']=Depreciation_byyear_PV
        df_curryear_PV['OM-PV']=OM_byyear_PV
        df_curryear_PV['TotalPower']=Total_power_byyear
        df_curryear_PV['SelfUseRevenue-PV']=SelfUse_revenue_PV
        df_curryear_PV['TransOutRevenue(UHV)-PV']=TransOut_revenue_PV_UHV
        df_curryear_PV['TransOutRevenue(CBS)-PV']=TransOut_revenue_PV_CBS
        df_curryear_PV['LCOE-PV']=LCOE_PV
        df_curryear_PV['LROE-PV']=LROE_PV
        df_curryear_PV['LROE-TransOut-PV']=LROE_TransOut_PV
        #df_CBStransfer_result=df_CBStransfer_result_list[idx_year]
        #df_CBStransfer_provincefrom=df_CBStransfer_result[df_CBStransfer_result['From']==province_from]
        #assert df_CBStransfer_provincefrom.shape[0]>0,"No Power Out From Province "+province_from+"!" 
        # 如果该省不需要跨省传输新能源，那么铁路的钱可以认为是省下来了，
        #average_distance=np.average(df_CBStransfer_provincefrom['Distance'],weights=df_CBStransfer_provincefrom['CBSPower'])
        #sum_CBSPower=np.sum(df_CBStransfer_provincefrom['CBSPower'])
        # 该年可以用到哪些年份生产的CBS呢？至多是前YR_usage_CBS年的
        df_curryear_PV_list.append(df_curryear_PV)
    #那么现在就可以给每个省份去计算，每一年下各个版本混合后的平均度电成本和度电收入
    LCOE_matrix=pd.DataFrame(columns=year_list,index=year_list)
    for idx_year,year in enumerate(year_list):
        LCOE_matrix.loc[year,df_curryear_PV_list[idx_year].index]=df_curryear_PV_list[idx_year]['LCOE-PV']
    LROE_matrix=pd.DataFrame(columns=year_list,index=year_list)
    for idx_year,year in enumerate(year_list):
        LROE_matrix.loc[year,df_curryear_PV_list[idx_year].index]=df_curryear_PV_list[idx_year]['LROE-PV']
    LROE_TransOut_matrix=pd.DataFrame(columns=year_list,index=year_list)
    for idx_year,year in enumerate(year_list):
        LROE_TransOut_matrix.loc[year,df_curryear_PV_list[idx_year].index]=df_curryear_PV_list[idx_year]['LROE-TransOut-PV']

    LCOE_matrix_byprovince_dict[province_from]=np.array(LCOE_matrix)
    LROE_matrix_byprovince_dict[province_from]=np.array(LROE_matrix)
    LROE_TransOut_matrix_byprovince_dict[province_from]=np.array(LROE_TransOut_matrix)
# 但是由于LCOE_matrix和LROE_matrix是基于光伏版本的矩阵，而不是储能柜版本的矩阵，所以不能与CBS相关的矩阵进行直接加减。
# 因而这里需要对LCOE_matrix和LROE_matrix做平均
df_LCOE_aggregated=pd.DataFrame(index=province_list,columns=year_list)
df_LROE_aggregated=pd.DataFrame(index=province_list,columns=year_list)
df_LROE_TransOut_aggregated=pd.DataFrame(index=province_list,columns=year_list)
for province in province_list:
    df_LCOE_aggregated.loc[province]=np.nanmean(LCOE_matrix_byprovince_dict[province],axis=0)
    df_LROE_aggregated.loc[province]=np.nanmean(LROE_matrix_byprovince_dict[province],axis=0)
    df_LROE_TransOut_aggregated.loc[province]=np.nanmean(LROE_TransOut_matrix_byprovince_dict[province],axis=0)

#%% 2.4 [储能柜]度电成本(不同储能柜版本、不含运输成本、不区分省份)
CBS_RentalCost_Matrix=pd.DataFrame(columns=year_list,index=year_list)
Whether_SecondLife=True
for idx_year,year in enumerate(year_list):
    # 为啥不考虑最后的回收利润呢？？？
    # 需不需要加上年折旧率？？？6%好像是？
    # 40尺对应的是20尺的2倍？（这个有影响吗）
    year_list_operation_CBS=np.arange(year,year+YR_usage_CBS)
    year_list_operation_CBS_until2060=np.arange(year,min(year+YR_usage_CBS,year_list[-1]+1))
    CBS_install_price=CBS_install_price_list[idx_year]
    CBS_install_price_without_VAT=CBS_install_price/(1+VAT_TAX)
    initial_capacity_CBS=Initial_Capacity_CBS_list[idx_year]
    # 第0年不会产生各种费用，所以需要np.insert一个0到第一年的成本项里
    OM_cost_list=np.insert(np.ones_like(year_list_operation_CBS)*OM_rate_list_CBS[idx_year],0,0.0)*CBS_install_price_without_VAT
    Loan_cost_list=np.insert(np.ones_like(year_list_operation_CBS)*Loan_rate_list_CBS[idx_year],0,0.0)*CBS_install_price
    #Admin_cost_list=np.insert(np.ones_like(year_list_operation_CBS)*Admin_rate_list_CBS[idx_year],0,0.0)*CBS_install_price_without_VAT
    Discount_rate_list=1/np.power(1+Discount_rate,np.arange(YR_usage_CBS+1))
    OM_cost_list=np.append(OM_cost_list,0.0)
    Loan_cost_list=np.append(Loan_cost_list,0.0)
    #Admin_cost_list=np.append(Admin_cost_list,0.0)
    Discount_rate_list=np.append(Discount_rate_list,Discount_rate_list[-1]/(1+Discount_rate))
    SecondLife_Revenue_list=np.zeros((len(Discount_rate_list),))
    if Whether_SecondLife:
        SecondLife_Revenue_list[-1]=CBS_install_price_without_VAT*RES_value_CBS*CBS_SecondLife_PriceRate
    InitialCost_list=np.zeros((len(Discount_rate_list),))
    InitialCost_list[0]=CBS_install_price_without_VAT
    # 如果考虑二阶段回收的话，需要对上述值都加入最后一年的回收收入
    Total_cost=np.inner(InitialCost_list+OM_cost_list+Loan_cost_list-SecondLife_Revenue_list,Discount_rate_list)
    #Average_cost_perrental=Total_cost/Rental_times_CBS_list[idx_year]
    #Average_price_perrental=Average_cost_perrental*(1+INCOME_TAX_CBS)*(1+Profit_rate_CBS)
    # 因为是度电成本，所以与年份无关，只与储能柜版本有关
    # 如果要加Discount Rate的话，下面的公式也能算但是会比较奇怪
    LCOE_Rental=Total_cost/np.inner(Capacity_matrix_CBS[idx_year,idx_year:idx_year+YR_usage_CBS],Discount_rate_list[1:-1])/1000/(Rental_times_CBS_list[idx_year]/YR_usage_CBS)*(1+INCOME_TAX_CBS)*(1+Profit_rate_CBS)
    # LCOE_Rental=Average_price_perrental/(initial_capacity_CBS*1000)#问题一堆啊，凭啥？？？但确实可以单纯作为简单的定价参考，所以可以不考虑实际运输不同储能的复杂情况
    CBS_RentalCost_Matrix.loc[year,year_list_operation_CBS_until2060]=LCOE_Rental
    # import sys
    # sys.exit(1)
CBS_RentalCost_Matrix=np.array(CBS_RentalCost_Matrix)
df_CBS_RentalCost_aggregated=pd.Series(np.nanmean(CBS_RentalCost_Matrix,axis=0),index=year_list)

#%% 2.5 [售电公司]利润
df_ServiceCost=df_LROE_TransOut_aggregated*ServiceFee_Rate

#%% 2.6 [最终发电公司]度电收入
df_LPOE_aggregated=pd.DataFrame(dtype=float,index=province_list,columns=year_list)
for province in province_list:
    df_LPOE_aggregated.loc[province]=np.array(df_LROE_aggregated.loc[province]-df_LCOE_aggregated.loc[province])-\
      np.array(Out_Rate_UHV_RenewableGen.loc[province])*np.array(df_UHV_LCOE_aggregated.loc[province])-\
      np.array(Out_Rate_CBS_RenewableGen.loc[province]+Out_Rate_UHV_RenewableGen.loc[province]*CoDeployment_Rate)*np.array(df_CBS_RentalCost_aggregated)-\
      np.array(Out_Rate_CBS_RenewableGen.loc[province])*np.array(df_LCOS_Railway_aggregated.loc[province])-\
      np.array(df_ServiceCost.loc[province]) # ServiceCost本身是针对TransOut部分做的抽成，所以无需乘上Out_Rate的系数

# 如果要考虑各省的LPOE均衡的话，就需要将LPOE根据该省新能源发电量进行均衡，因而需要对LPOE在df_renewable_generation之下进行均衡
df_RenewableXLPOE=df_renewable_generation*df_LPOE_aggregated
LPOE_AVG_weighted_byyear=np.sum(df_RenewableXLPOE,axis=0)/np.sum(df_renewable_generation,axis=0)
assert (LPOE_AVG_weighted_byyear>0).all(),"并非所有年份的利润均值均为正！"

df_LPOE_negative=df_LPOE_aggregated.copy()
df_LPOE_negative[df_LPOE_negative>0]=0
assert df_LPOE_negative.min().min()==0 # 不存在LPOE为负值的省份和年份

# 根据LPOE/LROE与电价/利润转移之间的计算公式，来推导利润转移所代表的省份间的等价收购电价变化
# df_ElectricityPrice_from_ProfitTransfer=df_ProfitTransfer*(1+VAT_TAX)/(1-ServiceFee_Rate)
# df_Electricity_PurchasePrice=df_ElectricityPrice_from_ProfitTransfer+TransOut_electricity_price
# df_Electricity_PurchasePrice=df_ElectricityPrice_from_ProfitTransfer

#%% 3.1 [绘制辅助]首先罗列中国地图的基本代码
province_cn2en_dict=dict(zip(province_basicinfo['Province-CN'],province_basicinfo.index))
file=r"../数据整理/中国省级地图GS（2019）1719号.geojson"
china_main=gpd.read_file(file).rename(columns={'CNAME':'省份'}).set_index('省份').drop(index=['香港','澳门','台湾'])
china_main['Province']=china_main.index.map(province_cn2en_dict)
china_main=china_main.reset_index().set_index('Province').loc[province_list]
china_main_crs=china_main.geometry.to_crs(epsg=2343).to_frame()
china_main_crs=china_main_crs.merge(province_basicinfo[['Lat','Lon','Region']],left_index=True,right_index=True)
china_main_crs['Lon-Center']=china_main_crs['geometry'].centroid.map(lambda t:t.x)
china_main_crs['Lat-Center']=china_main_crs['geometry'].centroid.map(lambda t:t.y)

province_coordinates=gpd.GeoDataFrame(province_basicinfo[['Lon','Lat']], geometry=gpd.points_from_xy(province_basicinfo['Lon'],province_basicinfo['Lat'],crs='EPSG:4326')) #EPSG就是经典的（经度，纬度）表示法
province_coordinates_converted=province_coordinates.geometry.to_crs(epsg=2343) # 替换成epsg中经典的表示法
xlim=(-2490128.154711554, 2968967.6930187442)
ylim=(1.8e6,6.1e6)
def get_xrel_yrel(ax_pos=mpl.transforms.Bbox([[0.12909373348498837,0.125],[0.8959062665150117,0.88]])):
    df_xrel_yrel=pd.DataFrame(0.0,index=province_list,columns=['x_abs','y_abs','x_rel_fig','y_rel_fig','x_rel_ax','y_rel_ax'])
    for province in province_list:
        x,y=china_main_crs.loc[province,'Lon-Center'],china_main_crs.loc[province,'Lat-Center']
        #x,y=province_coordinates_converted[province].x,province_coordinates_converted[province].y
        x_rel_fig=(x-xlim[0])/(xlim[1]-xlim[0])*(ax_pos.x1-ax_pos.x0)+ax_pos.x0 #省会相对于fig的位置（注意有别于相对于ax的位置）
        y_rel_fig=(y-ylim[0])/(ylim[1]-ylim[0])*(ax_pos.y1-ax_pos.y0)+ax_pos.y0
        x_rel_ax=(x-xlim[0])/(xlim[1]-xlim[0])
        y_rel_ax=(y-ylim[0])/(ylim[1]-ylim[0])
        df_xrel_yrel.loc[province,'x_abs']=x
        df_xrel_yrel.loc[province,'y_abs']=y
        df_xrel_yrel.loc[province,'x_rel_fig']=x_rel_fig
        df_xrel_yrel.loc[province,'y_rel_fig']=y_rel_fig
        df_xrel_yrel.loc[province,'x_rel_ax']=x_rel_ax
        df_xrel_yrel.loc[province,'y_rel_ax']=y_rel_ax
        
    return df_xrel_yrel
color_dict=dict({'Northern':'#dcccc8','Western':'#b9ebf0','Eastern':'#bfe2bf','Southern':'#d8d8d8','Central':'#bbd5e8'})
def darken_color(color, amount=2.5):
    # 如果amount大于1，那么就是darken_color,否则在0到1之间就是lighten_color
    import matplotlib.colors as mc
    import colorsys
    try:
        c=mc.cnames[color]
    except:
        c=color
    c=colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0],1-amount*(1-c[1]),c[2]) 

#%% 3.2 [绘制发电补贴]随年份和时间的变化
# 因为Subsidy绘制的都是相对值，所以没有必要去设置
year_list_withsubsidy=list(np.arange(year_list[0],year_list[0]+YR_Subsidy))
province_list_with_subsidy=list(df_Subsidy_PV.index[df_Subsidy_PV[year_list_withsubsidy].mean(axis=1)>0])
weighted_distance_list=np.zeros((len(province_transfer_profile_from),),dtype=float)
for idx_province,province in enumerate(province_list_with_subsidy):
    try:
        weighted_distance_list[idx_province]=province_transfer_profile_list_from[2].loc[province,'CBSPowerXDistance']/province_transfer_profile_list_from[2].loc[province,'CBSPower'] # 计算加权里程
    except:
        print(province) # 计算加权里程
# 查阅相关省份的平均送出电量

df_subsidy_coef=Out_Rate_CBS_RenewableGen[year_list_withsubsidy].mean(axis='columns')
china_main_crs['Subsidy-Coef']=df_subsidy_coef.replace(0.0,np.nan)

cmap=plt.get_cmap('viridis')
cmap_alpha=cmap(np.arange(cmap.N))
alpha_global=0.6
BG = np.asarray([1., 1., 1.,])
# Mix the colors with the background
for i in range(cmap.N):
    cmap_alpha[i,:-1]=cmap_alpha[i,:-1]*alpha_global+BG*(1-alpha_global)
cmap_alpha=mpl.colors.ListedColormap(cmap_alpha)


# 改为以地图形式绘制给予每个省份的补贴
fig,ax=plt.subplots(figsize=(10,8),dpi=100)
ax.set_ylim(ylim)
cbar_ax=fig.add_axes([0.8,0.2,0.03,0.4])
china_main_crs.plot(ax=ax,edgecolor='black',facecolor='white')
china_main_crs.plot('Subsidy-Coef',cmap=cmap_alpha,ax=ax,legend=True,cax=cbar_ax,edgecolor='black',vmin=0.1,vmax=1.0)
ax.annotate('CBS Power \n Proportion',xy=(0.62,0.065),xycoords='figure fraction')
ax.annotate('Subsidy',xy=(0.11,0.71),xycoords='figure fraction',color='#E07117',weight='bold')
max_subsidy=df_Subsidy_PV.max().max()
ax_pos=ax.get_position()
for province in province_list_with_subsidy:
    df_xrel_yrel=get_xrel_yrel(ax_pos)
    x_rel,y_rel=df_xrel_yrel.loc[province,'x_rel_fig'],df_xrel_yrel.loc[province,'y_rel_fig']
    bar_ax=fig.add_axes([x_rel-0.02,y_rel,0.06,0.1])
    bar_ax.set_ylim([0,max_subsidy])
    bar_ax.bar(np.arange(YR_Subsidy),df_Subsidy_PV.loc[province,year_list_withsubsidy],color='orange',edgecolor='black')
    bar_ax.set_axis_off()
ax.axis('off') 
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
fig.savefig('图片/Fig4-1-发电补贴.pdf',format='pdf',bbox_inches='tight')

#%% 3.2 [绘制铁路成本的分布地图]
fig,ax=plt.subplots(figsize=(10,8),dpi=100)
ax.set_ylim(ylim)
cbar_ax=fig.add_axes([0.8,0.2,0.03,0.4])
curr_year=2031
province_list_withRailway_curryear=Out_Rate_CBS_RenewableGen[Out_Rate_CBS_RenewableGen[curr_year]==0.0].index
LCOS_Railway_curryear=df_LCOS_Railway_aggregated[curr_year].copy()
LCOS_Railway_curryear.loc[province_list_withRailway_curryear]=np.nan
china_main_crs['LCOS_Railway']=LCOS_Railway_curryear
china_main_crs.plot(ax=ax,edgecolor='black',facecolor='white')
china_main_crs.plot('LCOS_Railway',ax=ax,cmap=cmap_alpha,legend=True,cax=cbar_ax,edgecolor='black')
ax.annotate('Railway Cost\n (US$/kWh)',xy=(0.59,0.068),xycoords='figure fraction')
ax.axis('off') 

railway_cost_adjustment_total=(weights_Railway_yearAvg*df_LCOS_Railway_adjustment)[curr_year]
province_list_with_adjustment=railway_cost_adjustment_total[railway_cost_adjustment_total!=0.0].index

ax_pos=ax.get_position()
xmin,xmax=ax.get_xlim()
ymin,ymax=ax.get_ylim()
max_railway_adjustment=railway_cost_adjustment_total.abs().max()
for province in province_list_with_adjustment:
    df_xrel_yrel=get_xrel_yrel(ax_pos)
    x_rel_ax,y_rel_ax=df_xrel_yrel.loc[province,'x_rel_ax'],df_xrel_yrel.loc[province,'y_rel_ax']
    if railway_cost_adjustment_total.loc[province]>0:
        ax.arrow(x_rel_ax*(xmax-xmin)+xmin,y_rel_ax*(ymax-ymin)+ymin,0.0,
                 railway_cost_adjustment_total.loc[province]/max_railway_adjustment*(ymax-ymin)*0.15,
                 color='brown',width=30000)
    else:
        ax.arrow(x_rel_ax*(xmax-xmin)+xmin,y_rel_ax*(ymax-ymin)+ymin,0.0,
                 railway_cost_adjustment_total.loc[province]/max_railway_adjustment*(ymax-ymin)*0.15,
                 color='blue',width=30000)

from matplotlib.lines import Line2D
# 添加对于红蓝色箭头的标注
ax_pos=ax.get_position()
ax.arrow(x=-1773105,y=2153587,dx=0.0,dy=99977,color='brown',width=30000)
ax.arrow(x=-1773105,y=2053609,dx=0.0,dy=-99977,color='blue',width=30000)
ax.annotate('Total Cost Increase',xy=(-1673105,2153587+149977*0.3),color='brown')
ax.annotate('Total Cost Decrease',xy=(-1673105,2053609-149977*1.2),color='#00008B')
# ax.arrow(x=arrow_legend_x,y=arrow_legend_y1,dx=0.0,dy=arrow_legend_length,color='red',width=30000)
# ax.arrow(x=arrow_legend_x,y=arrow_legend_y2,dx=0.0,dy=-arrow_legend_length,color='blue',width=30000)
# red_arrow_patch=Patch(color='white',label='Total Cost Increase',alpha=0)
# blue_arrow_patch=Patch(color='white',label='Total Cost Decrease',alpha=0)

# # red_arrow_patch=Line2D(color='red')
# # blue_arrow_patch=Line2D(color='blue')
# leg=ax.legend(handles=[red_arrow_patch,blue_arrow_patch],loc='lower left',edgecolor='None',framealpha=0,bbox_to_anchor=(0.0,-0.05))
# legend_window=leg.get_window_extent()
# fig_size=fig.get_size_inches()*fig.dpi
# ax_pos=ax.get_position()
# legend_xmin_axpos=(legend_window.p0[0]/fig_size[0]-ax_pos.x0)/(ax_pos.x1-ax_pos.x0)*(xmax-xmin)+xmin
# legend_xmax_axpos=(legend_window.p1[0]/fig_size[0]-ax_pos.x0)/(ax_pos.x1-ax_pos.x0)*(xmax-xmin)+xmin
# legend_ymin_axpos=(legend_window.p0[1]/fig_size[1]-ax_pos.y0)/(ax_pos.y1-ax_pos.y0)*(ymax-ymin)+ymin
# legend_ymax_axpos=(legend_window.p1[1]/fig_size[1]-ax_pos.y0)/(ax_pos.y1-ax_pos.y0)*(ymax-ymin)+ymin
# arrow_legend_x=legend_xmin_axpos+(legend_xmax_axpos-legend_xmin_axpos)*0.2
# arrow_legend_y1=legend_ymin_axpos+(legend_ymax_axpos-legend_ymin_axpos)*0.6
# arrow_legend_y2=legend_ymin_axpos+(legend_ymax_axpos-legend_ymin_axpos)*0.4
# arrow_legend_length=(legend_ymax_axpos-legend_ymin_axpos)*0.2
# ax.arrow(x=arrow_legend_x,y=arrow_legend_y1,dx=0.0,dy=arrow_legend_length,color='red',width=30000)
# ax.arrow(x=arrow_legend_x,y=arrow_legend_y2,dx=0.0,dy=-arrow_legend_length,color='blue',width=30000)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
fig.savefig('图片/Fig4-2-铁路成本.pdf',format='pdf',bbox_inches='tight')

#%% 3.4 [发电公司利润拆解]（采用stacked bar plot来设计）
# 我们以内蒙古和广东两个电能输入输出大省来进行分析
from matplotlib import cm
#colors=[cm.jet(x) for x in np.linspace(1.0,0.8,4)]
colors=['#AEB2D1','#D9B9D4','#92A5D1','gray','#B2D3A4']#'#C5DFF4'
# 注意这些东西都需要加上Price Transfer，除以Exchange Rate，以及
fig,ax=plt.subplots(3,1,figsize=(10,12),tight_layout=True)
alpha_global=1.0
curr_province='Guangdong'
# 广东并不存在Railway Cost和Subsidy
Total_Revenue=np.array(df_LROE_aggregated.loc[curr_province])/Exchange_rate
PV_cost=np.array(df_LCOE_aggregated.loc[curr_province])/Exchange_rate
UHV_DeliverCost=np.array(Out_Rate_UHV_RenewableGen.loc[curr_province]*df_UHV_LCOE_aggregated.loc[curr_province])/Exchange_rate
CBS_RentalCost=np.array(Out_Rate_CBS_RenewableGen.loc[curr_province]+Out_Rate_UHV_RenewableGen.loc[curr_province]*CoDeployment_Rate)*np.array(df_CBS_RentalCost_aggregated)/Exchange_rate
Service_Cost=np.array(df_ServiceCost.loc[curr_province])/Exchange_rate
LPOE=Total_Revenue-CBS_RentalCost-Service_Cost-PV_cost
ax[0].bar(year_list,PV_cost,color=colors[0],alpha=alpha_global)
ax[0].bar(year_list,CBS_RentalCost,bottom=PV_cost,color=colors[2],alpha=alpha_global)
ax[0].bar(year_list,Service_Cost+UHV_DeliverCost,bottom=PV_cost+CBS_RentalCost,color=colors[3],alpha=alpha_global)
ax[0].bar(year_list,LPOE,bottom=PV_cost+CBS_RentalCost+Service_Cost+UHV_DeliverCost,color=colors[4],alpha=alpha_global)
ax[0].annotate(curr_province+' ('+province_basicinfo.loc[curr_province,'Region']+')',xy=(0.15,0.85),xycoords=('axes fraction'))
ax[0].set_ylim([0,0.25])
ax[0].set_xticks([])
# ax[0].set_ylabel('Unit Price ($/kWh)')

curr_province='InnerMongolia'
Total_Revenue=np.array(df_LROE_aggregated.loc[curr_province])/Exchange_rate 
Subsidy=np.array(df_Subsidy_PV.loc[curr_province])/Exchange_rate
PV_cost=np.array(df_LCOE_aggregated.loc[curr_province])/Exchange_rate
UHV_DeliverCost=np.array(Out_Rate_UHV_RenewableGen.loc[curr_province]*df_UHV_LCOE_aggregated.loc[curr_province])/Exchange_rate
CBS_RailwayCost=np.array(Out_Rate_CBS_RenewableGen.loc[curr_province]*df_LCOS_Railway_aggregated.loc[curr_province])/Exchange_rate
CBS_RentalCost=np.array(Out_Rate_CBS_RenewableGen.loc[curr_province]+Out_Rate_UHV_RenewableGen.loc[curr_province]*CoDeployment_Rate)*np.array(df_CBS_RentalCost_aggregated)/Exchange_rate
Service_Cost=np.array(df_ServiceCost.loc[curr_province])/Exchange_rate
LPOE=Total_Revenue-CBS_RailwayCost-CBS_RentalCost-Service_Cost-PV_cost
ax[1].bar(year_list,PV_cost,color=colors[0],alpha=alpha_global)
ax[1].bar(year_list,CBS_RentalCost,bottom=PV_cost,facecolor=colors[1],alpha=alpha_global)
ax[1].bar(year_list,CBS_RailwayCost,bottom=PV_cost+CBS_RentalCost,facecolor=colors[2],alpha=alpha_global)
ax[1].bar(year_list,Service_Cost+UHV_DeliverCost,bottom=PV_cost+CBS_RailwayCost+CBS_RentalCost,facecolor=colors[3],alpha=alpha_global)
ax[1].bar(year_list,LPOE,bottom=PV_cost+CBS_RailwayCost+CBS_RentalCost+Service_Cost+UHV_DeliverCost,facecolor=colors[4],alpha=alpha_global)
ax[1].bar(year_list,Subsidy,bottom=PV_cost+CBS_RailwayCost+CBS_RentalCost+Service_Cost+UHV_DeliverCost+LPOE-Subsidy, facecolor='none',hatch='///',alpha=alpha_global)
ax[1].annotate(curr_province+' ('+province_basicinfo.loc[curr_province,'Region']+')',xy=(0.15,0.85),xycoords=('axes fraction'))
ax[1].set_ylim([0,0.25])
ax[1].set_xticks([])
# ax[1].set_ylabel('Unit Price ($/kWh)')

curr_province='Xinjiang'
Total_Revenue=np.array(df_LROE_aggregated.loc[curr_province])/Exchange_rate 
Subsidy=np.array(df_Subsidy_PV.loc[curr_province])/Exchange_rate
PV_cost=np.array(df_LCOE_aggregated.loc[curr_province])/Exchange_rate
UHV_DeliverCost=np.array(Out_Rate_UHV_RenewableGen.loc[curr_province]*df_UHV_LCOE_aggregated.loc[curr_province])/Exchange_rate
CBS_RailwayCost=np.array(Out_Rate_CBS_RenewableGen.loc[curr_province]*df_LCOS_Railway_aggregated.loc[curr_province])/Exchange_rate
CBS_RentalCost=np.array(Out_Rate_CBS_RenewableGen.loc[curr_province]+Out_Rate_UHV_RenewableGen.loc[curr_province]*CoDeployment_Rate)*np.array(df_CBS_RentalCost_aggregated)/Exchange_rate
Service_Cost=np.array(df_ServiceCost.loc[curr_province])/Exchange_rate
LPOE=Total_Revenue-CBS_RailwayCost-CBS_RentalCost-Service_Cost-PV_cost
ax[2].bar(year_list,PV_cost,color=colors[0],label='PV Cost',alpha=alpha_global)
ax[2].bar(year_list,CBS_RentalCost,bottom=PV_cost,facecolor=colors[1],label='CBS Rental Cost',alpha=alpha_global)
ax[2].bar(year_list,CBS_RailwayCost,bottom=PV_cost+CBS_RentalCost,facecolor=colors[2],label='CBS Transportation Cost',alpha=alpha_global)
ax[2].bar(year_list,Service_Cost+UHV_DeliverCost,bottom=PV_cost+CBS_RailwayCost+CBS_RentalCost,facecolor=colors[3],label='Other Costs',alpha=alpha_global)
ax[2].bar(year_list,LPOE,bottom=PV_cost+CBS_RailwayCost+CBS_RentalCost+Service_Cost+UHV_DeliverCost,facecolor=colors[4],label='Profit',alpha=alpha_global)
ax[2].bar(year_list,Subsidy,bottom=PV_cost+CBS_RailwayCost+CBS_RentalCost+Service_Cost+UHV_DeliverCost+LPOE-Subsidy, facecolor='none',hatch='///',label='Subsidy',alpha=alpha_global)
ax[2].annotate(curr_province+' ('+province_basicinfo.loc[curr_province,'Region']+')',xy=(0.15,0.85),xycoords=('axes fraction'))
ax[2].set_ylim([0,0.25])
ax[2].set_xticks([])
# ax[2].set_ylabel('Unit Price ($/kWh)')
ax[2].legend(bbox_to_anchor=(1.05,-0.1),ncol=2)
fig.supylabel('                   Unit Price (US$/kWh)')
fig.patch.set_alpha(0.0)
ax[0].patch.set_alpha(0.0)
ax[1].patch.set_alpha(0.0)
ax[2].patch.set_alpha(0.0)
fig.savefig('图片/Fig4-3-发电公司利润（代表省份）.pdf',format='pdf',bbox_inches='tight')

#%% 3.5 [绘制发电公司利润地图]
china_main_crs_with_LPOE=china_main_crs.merge(df_LPOE_aggregated/Exchange_rate,left_index=True,right_index=True)
fig,ax=plt.subplots(2,2,figsize=(15,15))
cbar_ax=fig.add_axes([0.9,0.1,0.05,0.5])
vmin=0.0/Exchange_rate
vmax=df_LPOE_aggregated.max().max()/Exchange_rate
curr_year=2025
ax[0,0].set_ylim(ylim)
china_main_crs_with_LPOE.plot(curr_year,ax=ax[0,0],cmap=cmap_alpha,edgecolor='black',vmin=vmin,vmax=vmax,legend=True,cax=cbar_ax,legend_kwds={'label':'Unit Profit (US$/kWh)'})
ax[0,0].axis('off')
ax[0,0].annotate(str(curr_year),xy=(0.45,0.8),xycoords=('axes fraction'),weight='bold',fontsize=30)

curr_year=2031
ax[0,1].set_ylim(ylim)
china_main_crs_with_LPOE.plot(curr_year,ax=ax[0,1],cmap=cmap_alpha,edgecolor='black',vmin=vmin,vmax=vmax)
ax[0,1].axis('off')
ax[0,1].annotate(str(curr_year),xy=(0.45,0.8),xycoords=('axes fraction'),weight='bold',fontsize=30)

curr_year=2046
ax[1,0].set_ylim(ylim)
china_main_crs_with_LPOE.plot(curr_year,ax=ax[1,0],cmap=cmap_alpha,edgecolor='black',vmin=vmin,vmax=vmax)
ax[1,0].axis('off')
ax[1,0].annotate(str(curr_year),xy=(0.45,0.8),xycoords=('axes fraction'),weight='bold',fontsize=30)

curr_year=2060
ax[1,1].set_ylim(ylim)
china_main_crs_with_LPOE.plot(curr_year,ax=ax[1,1],cmap=cmap_alpha,edgecolor='black',vmin=vmin,vmax=vmax)

ax[1,1].axis('off')
ax[1,1].annotate(str(curr_year),xy=(0.45,0.8),xycoords=('axes fraction'),weight='bold',fontsize=30)
fig.tight_layout(rect=[0,0,0.93,0.7])
fig.patch.set_alpha(0.0)
ax[0,0].patch.set_alpha(0.0)
ax[0,1].patch.set_alpha(0.0)
ax[1,0].patch.set_alpha(0.0)
ax[1,1].patch.set_alpha(0.0)
fig.savefig('图片/Fig4-4-发电公司利润（全国）.pdf',format='pdf',bbox_inches='tight')

#%% 3.6 [绘制储能公司利润]（毕竟储能公司的利润是不区分省份的）
# 这里画成和Service Cost一样的图片
CBS_Profit_byyear=np.nanmean(np.array(CBS_RentalCost_Matrix,dtype=float)/(1+Profit_rate_CBS)*Profit_rate_CBS/Exchange_rate,axis=0)
CBS_Profit_NationalSum=CBS_Profit_byyear*np.array(df_renewable_generation.sum(axis='index'))/Exchange_rate
fig,ax=plt.subplots(figsize=(10,5),tight_layout=True)
color1='#2e7ebb'
color2='#4F845C'
ax.bar(year_list,CBS_Profit_NationalSum,color=color1,width=0.6,alpha=0.45)
ax.tick_params(axis='y', colors=color1)
#ax.bar(year_list+0.3,(df_LPOE_aggregated*df_renewable_generation).sum(axis='index'),width=0.3,color='brown')
ax2=ax.twinx()
ax2.plot(year_list,CBS_Profit_byyear,'^-',color=color2,lw=2,markersize=8)
ax2.tick_params(axis='y', colors=color2)
ax.set_xlabel('Year')
ax.set_ylabel('Total Profit\n(Billion US$)',color=color1)
ax2.set_ylabel('Unit Profit (US$/kWh)',color=color2)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax2.ticklabel_format(axis='y',style='sci',scilimits=(0,1))
fig.savefig('图片/Fig4-5-储能公司利润.pdf',format='pdf',bbox_inches='tight')

#%% 3.7 [绘制售电公司利润]
#Service_Cost_National=ServiceProvision_Charge.sum(axis='index')/Exchange_rate
Service_Cost_NationalSum=(df_ServiceCost*df_renewable_generation).sum(axis='index')/Exchange_rate
Service_Cost_NationalAvg=Service_Cost_NationalSum/df_renewable_generation.sum(axis='index')
fig,ax=plt.subplots(figsize=(10,5),tight_layout=True)
color1='#2e7ebb'
color2='#4F845C'#'#7262ac'
ax.bar(year_list,Service_Cost_NationalSum,color=color1,width=0.6,alpha=0.45)
ax.tick_params(axis='y', colors=color1)
#ax.bar(year_list+0.3,(df_LPOE_aggregated*df_renewable_generation).sum(axis='index'),width=0.3,color='brown')
ax2=ax.twinx()

ax2.plot(Service_Cost_NationalAvg.index,Service_Cost_NationalAvg,'^-',color=color2,lw=2,markersize=8)
#ax2.spines['right'].set_color(color2)
#ax2.spines['left'].set_color(color2)
ax2.tick_params(axis='y', colors=color2)

ax.set_xlabel('Year')
ax.set_ylabel('Total Profit\n(Billion US$)',color=color1)
ax2.set_ylabel('Unit Profit (US$/kWh)',color=color2)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax2.ticklabel_format(axis='y',style='sci',scilimits=(0,1))
fig.savefig('图片/Fig4-6-售电公司利润.pdf',format='pdf',bbox_inches='tight')
#%% 3.8 [绘制储能公司利润分布]的构成饼图
# 由于饼图只看比例不看绝对值，所以按储能公司成本来计算也是非常正常的
df_CBS_profit=(Out_Rate_CBS_RenewableGen+Out_Rate_UHV_RenewableGen*CoDeployment_Rate)*pd.DataFrame(np.repeat(np.array(df_CBS_RentalCost_aggregated)[None,:],num_province,axis=0),index=province_list,columns=year_list)
CBS_Rental_Profit_aggregated=(df_CBS_profit*df_renewable_generation).groupby(province_basicinfo['Region'].to_dict()).agg(sum)
fig,ax=plt.subplots(1,4,figsize=(15,15))
for idx_curr_year,curr_year in enumerate([2030,2040,2050,2060]):
    ax[idx_curr_year].pie(CBS_Rental_Profit_aggregated[curr_year],colors=[darken_color(color_dict[region]) for region in CBS_Rental_Profit_aggregated.index])
    ax[idx_curr_year].set_title(str(curr_year))
    ax[idx_curr_year].patch.set_alpha(0.0)

legend_handle_list=[]
for idx_region,region in enumerate(CBS_Rental_Profit_aggregated.index):
    legend_handle_list.append(Patch(color=darken_color(color_dict[region]),label=region))
ax[1].legend(handles=legend_handle_list,ncol=5,bbox_to_anchor=(4,0))
fig.patch.set_alpha(0.0)
#fig.savefig('图片/Fig4-7-储能公司利润（饼图）.pdf',format='pdf',bbox_inches='tight')

#%% 3.9 [绘制售电公司利润分布]的构成饼图
ServiceProvision_Charge_aggregated=(df_ServiceCost*df_renewable_generation).groupby(province_basicinfo['Region'].to_dict()).agg(sum)
fig,ax=plt.subplots(1,4,figsize=(15,15))
for idx_curr_year,curr_year in enumerate([2030,2040,2050,2060]):
    ax[idx_curr_year].pie(ServiceProvision_Charge_aggregated[curr_year],colors=[darken_color(color_dict[region]) for region in ServiceProvision_Charge_aggregated.index])
    ax[idx_curr_year].set_title(str(curr_year))
    ax[idx_curr_year].patch.set_alpha(0.0)

legend_handle_list=[]
for idx_region,region in enumerate(ServiceProvision_Charge_aggregated.index):
    legend_handle_list.append(Patch(color=darken_color(color_dict[region]),label=region))
ax[1].legend(handles=legend_handle_list,ncol=5,bbox_to_anchor=(4,0))
fig.patch.set_alpha(0.0)
#fig.savefig('图片/Fig4-8-售电公司利润（饼图）.pdf',format='pdf',bbox_inches='tight')

#%% 测算一下各年份的df_LPOE_aggregated
LPOE_weightedAvg_byyear=pd.Series(index=year_list)
for year in year_list:
    weights=df_renewable_generation[year]
    LPOE_weightedAvg_byyear.loc[year]=np.average(df_LPOE_aggregated[year],weights=weights)
    
#%% 如何查看Energy Arbitrage情况
curr_year=2031
idx_curr_year=curr_year-year_list[0]
df_CBStransfer_curryear=df_CBStransfer_result_list[idx_curr_year]
df_arbitrage=df_CBStransfer_curryear.copy()
df_arbitrage['Arbitrage']=0.0
for idx in df_CBStransfer_curryear.index:
    province_from=df_CBStransfer_curryear.loc[idx,'From']
    province_to=df_CBStransfer_curryear.loc[idx,'To']
    df_arbitrage.loc[idx,'Arbitrage']=SelfUse_electricity_price[province_list.index(province_to),idx_year]-SelfUse_electricity_price[province_list.index(province_from),idx_year]
df_arbitrage['Arbitrage*CBSPower']=df_arbitrage['Arbitrage']*df_arbitrage['CBSPower']
df_arbitrage=df_arbitrage.groupby('From').agg({'Arbitrage*CBSPower':sum,'CBSPower':sum})
df_arbitrage['Weighted']=df_arbitrage['Arbitrage*CBSPower']/df_arbitrage['CBSPower']







#%% 4.1根据张璇老师的建议，添加关于交易总量的信息
# provincial_distance=pd.read_excel(excel_name,sheet_name='3.ProvincialDistance',usecols='A:AF',index_col=0)

# df_result_CBS_positive_list=[]
# df_result_UHV_positive_list=[]
# for curr_year in year_list:
#     # 找到那个找最优解的程序
#     model=gp.Model("NetworkFlow_CBS")
#     model.Params.LogToConsole=0
#     # 考察以CBS方式运送煤炭的数量及最优化方法
#     df_transfer_curryear=df_transfer_byyear_list[curr_year-year_list[0]]
#     province_out=list(df_transfer_curryear[df_transfer_curryear['RenewableOut']-df_transfer_curryear['RenewableOut(UHV)']<0].index)
#     province_in=list(df_transfer_curryear[df_transfer_curryear['RenewableIn']-df_transfer_curryear['RenewableIn(UHV)']>0].index)
#     province_combination=[(i,j) for i in province_out for j in province_in]
    
#     province_flow=model.addVars(province_combination,lb=0.0,vtype=GRB.CONTINUOUS)
#     model.addConstrs((province_flow.sum(i,"*")==-(df_transfer_curryear.loc[i,'RenewableOut']-df_transfer_curryear.loc[i,'RenewableOut(UHV)']) for i in province_out),"FlowOut")
#     model.addConstrs((province_flow.sum("*",j)==(df_transfer_curryear.loc[j,'RenewableIn']-df_transfer_curryear.loc[j,'RenewableIn(UHV)']) for j in province_in),"FlowIn")
#     obj=sum(province_flow[(i,j)]*provincial_distance.loc[i,j] for (i,j) in province_combination)
#     model.setObjective(obj)
#     model.optimize()
#     df_result_CBS=pd.DataFrame(columns=['From','To','CBSAmount'])
#     for idx,(i,j) in enumerate(province_combination):
#         df_result_CBS.loc[idx]=[i,j,province_flow[(i,j)].X]
#     df_result_CBS_positive=df_result_CBS[df_result_CBS['CBSAmount']>0].reset_index(drop=True)
#     df_result_CBS_positive_list.append(df_result_CBS_positive)
    
#     province_out_UHV=list(df_transfer_curryear[df_transfer_curryear['RenewableOut(UHV)']<0].index)
#     province_in_UHV=list(df_transfer_curryear[df_transfer_curryear['RenewableIn(UHV)']>0].index)
#     province_combination_UHV=[(i,j) for i in province_out_UHV for j in province_in_UHV]
#     model_UHV=gp.Model("NetworkFlow_UHV")
#     model_UHV.Params.LogToConsole=0
#     province_flow_UHV=model_UHV.addVars(province_combination_UHV,lb=0.0,vtype=GRB.CONTINUOUS)
#     model_UHV.addConstrs((province_flow_UHV.sum(i,"*")==-df_transfer_curryear.loc[i,'RenewableOut(UHV)'] for i in province_out_UHV),"FlowOut")
#     model_UHV.addConstrs((province_flow_UHV.sum("*",j)==df_transfer_curryear.loc[j,'RenewableIn(UHV)'] for j in province_in_UHV),"FlowIn")
#     obj_UHV=sum(province_flow_UHV[(i,j)]*provincial_distance.loc[i,j] for (i,j) in province_combination_UHV)
#     model_UHV.setObjective(obj_UHV)
#     model_UHV.optimize()
#     df_result_UHV=pd.DataFrame(columns=['From','To','UHVAmount'])
#     for idx,(i,j) in enumerate(province_combination_UHV):
#         df_result_UHV.loc[idx]=[i,j,province_flow_UHV[(i,j)].X]
#     df_result_UHV_positive=df_result_UHV[df_result_UHV['UHVAmount']>0].reset_index(drop=True)
#     df_result_UHV_positive_list.append(df_result_UHV_positive)

#%% 4.2关于Interprovincial profits的资金流分析
# 单位：由于电量单位是TWh，而电价的单位是¥/kWh
# 首先分析从用电公司向售电公司提交的成本
df_cashflow_consumption2sales=pd.DataFrame(0.0,index=province_list,columns=year_list)
df_cashflow_sales2generation=pd.DataFrame(0.0,index=province_list,columns=year_list)
df_CBS_transfer_from=pd.DataFrame(0.0,index=province_list,columns=year_list)

for idx_curr_year,curr_year in enumerate(year_list):
    #df_UHV=df_result_UHV_positive_list[idx_curr_year]
    df_CBS=df_CBStransfer_result_list[idx_curr_year]
    # for idx_UHV in df_UHV.index:
    #     province_from=df_UHV.loc[idx_UHV,'From']
    #     province_to=df_UHV.loc[idx_UHV,'To']
    #     df_cashflow_consumption2sales.loc[province_to,curr_year]+=df_UHV.loc[idx_UHV,'UHVAmount']*TransOut_electricity_price[province_list.index(province_to),idx_curr_year]
    #     df_cashflow_sales2generation.loc[province_from,curr_year]+=df_UHV.loc[idx_UHV,'UHVAmount']*TransOut_electricity_price[province_list.index(province_to),idx_curr_year]
    for idx_CBS in df_CBS.index:
        province_from=df_CBS.loc[idx_CBS,'From']
        province_to=df_CBS.loc[idx_CBS,'To']
        df_cashflow_consumption2sales.loc[province_to,curr_year]+=df_CBS.loc[idx_CBS,'CBSPower']*TransOut_electricity_price[province_list.index(province_to),idx_curr_year]
        df_cashflow_sales2generation.loc[province_from,curr_year]+=df_CBS.loc[idx_CBS,'CBSPower']*TransOut_electricity_price[province_list.index(province_to),idx_curr_year]
        df_CBS_transfer_from.loc[province_from,curr_year]+=df_CBS.loc[idx_CBS,'CBSPower']
df_cashflow_sales2generation*=1-ServiceFee_Rate

# 发电公司→储能公司
df_cashflow_generation2storage=pd.DataFrame(0.0,index=province_list,columns=year_list)
for curr_year in year_list:
    df_cashflow_generation2storage[curr_year]=df_CBS_transfer_from[curr_year]*df_CBS_RentalCost_aggregated.loc[curr_year]

# 发电公司→铁路公司
df_cashflow_generation2railway=df_LCOS_Railway_aggregated*df_CBS_transfer_from

# 发电公司→光伏/风电公司
df_cashflow_generation2PV=df_LCOE_aggregated*df_CBS_transfer_from

# 政府→发电公司
df_cashflow_subsidy=df_Subsidy_PV*df_CBS_transfer_from

#%% 画图-柱状图
fig,ax=plt.subplots(figsize=(10,5))
#years_of_interest=[2025,2031,2046,2060]
years_of_interest=year_list
idx_years_of_interest=np.arange(len(years_of_interest))
consumption2sales=df_cashflow_consumption2sales.sum(axis='index')[year_list]/Exchange_rate/1e3
sales2generation=df_cashflow_sales2generation.sum(axis='index')[year_list]/Exchange_rate/1e3
generation2storage=df_cashflow_generation2storage.sum(axis='index')[year_list]/Exchange_rate/1e3
generation2railway=df_cashflow_generation2railway.sum(axis='index')[year_list]/Exchange_rate/1e3
generation2PV=df_cashflow_generation2PV.sum(axis='index')[year_list]/Exchange_rate/1e3
ax.bar(years_of_interest,consumption2sales,label='Consumption→Sales',color='#237EA8',alpha=alpha_global)
ax.bar(years_of_interest,sales2generation,bottom=consumption2sales,label='Sales→Generation',color='#36AAE2',alpha=alpha_global)
ax.bar(years_of_interest,generation2railway,bottom=consumption2sales+sales2generation,label='Generation→Transportation',color='#808080',alpha=alpha_global)
ax.bar(years_of_interest,generation2storage,bottom=consumption2sales+sales2generation+generation2railway,label='Generation→Storage',color='green',alpha=alpha_global)
ax.bar(years_of_interest,generation2PV,bottom=consumption2sales+sales2generation+generation2railway+generation2storage,label='Generation→Generator',color='#C76E00',alpha=alpha_global)
ax.set_ylabel('Total Transaction\n(Trillion US$)')
ax.set_xlabel('Year')
ax.set_ylim([0,1])
ax.legend(bbox_to_anchor=(1.17,-0.15),ncol=2)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
fig.savefig('图片/Fig3-2-总交易量.pdf',format='pdf',bbox_inches='tight')

# ax.bar(idx_years_of_interest-0.3,consumption2sales,width=0.15)
# ax.bar(idx_years_of_interest-0.15,sales2generation,width=0.15)
# ax.bar(idx_years_of_interest,generation2railway,width=0.15)
# ax.bar(idx_years_of_interest+0.15,generation2PV,width=0.15)
# ax.bar(idx_years_of_interest+0.3,generation2railway,width=0.15)
#ax.set_xticks(idx_years_of_interest)

#%%画图-各省份图
transaction_byprovince=(df_cashflow_consumption2sales+df_cashflow_sales2generation+df_cashflow_generation2storage+df_cashflow_generation2railway+df_cashflow_generation2PV).mean(axis='columns')/Exchange_rate/1e3
transaction_byprovince=transaction_byprovince.astype(float)
transaction_byprovince.name='Transaction'
china_main_crs_with_transaction=china_main_crs.merge(transaction_byprovince,left_index=True,right_index=True)
fig,ax=plt.subplots(figsize=(15,15))
cbar_ax=fig.add_axes([0.9,0.25,0.02,0.5])
vmin=0.0/Exchange_rate
vmax=transaction_byprovince.max()
ax.set_ylim(ylim)
china_main_crs_with_transaction.plot('Transaction',cmap=cmap_alpha,ax=ax,edgecolor='black',vmin=vmin,vmax=vmax,legend=True,cax=cbar_ax,legend_kwds={'label':'Total Transaction (Trillion US$)'})
ax.axis('off')
fig.savefig('图片/Fig3-3-交易分布.pdf',format='pdf',bbox_inches='tight')
# transaction_byprovince=(df_cashflow_consumption2sales+df_cashflow_sales2generation+df_cashflow_generation2storage+df_cashflow_generation2railway+df_cashflow_generation2PV).mean(axis='columns')/Exchange_rate/1e3
# transaction_byprovince=transaction_byprovince.astype(float)
# transaction_byprovince.name='Transaction'
# china_main_crs_with_transaction=china_main_crs.merge(transaction_byprovince,left_index=True,right_index=True)
# fig,ax=plt.subplots(figsize=(15,15))
# cbar_ax=fig.add_axes([0.9,0.25,0.02,0.5])
# vmin=0.0/Exchange_rate
# vmax=transaction_byprovince.max()
# ax.set_ylim(ylim)
# china_main_crs_with_transaction.plot('Transaction',cmap=cmap_alpha,ax=ax,edgecolor='black',vmin=vmin,vmax=vmax,legend=True,cax=cbar_ax,legend_kwds={'label':'Total Transaction (Trillion US$)'})
# ax.axis('off')
# fig.savefig('图片/Fig3-3-交易分布.pdf',format='pdf',bbox_inches='tight')

#%% 如果换成heatmap会不会更好些？
# df_cashflow_total=(df_cashflow_consumption2sales+df_cashflow_sales2generation+df_cashflow_generation2storage+df_cashflow_generation2railway+df_cashflow_generation2PV).astype(float)/Exchange_rate/1e3
# df_cashflow_total['Region']=province_basicinfo['Region']
# df_cashflow_total.sort_values(by='Region',inplace=True)
# df_cashflow_total.pop('Region')
# fig,ax=plt.subplots(figsize=(12,20))
# sns.heatmap(df_cashflow_total)


#%% 如果换成多个图会不会好看一些
# ax.bar(years_of_interest,consumption2sales,label='Consumption→Sales',color='#237EA8',alpha=alpha_global)
# ax.bar(years_of_interest,sales2generation,bottom=consumption2sales,label='Sales→Generation',color='#36AAE2',alpha=alpha_global)
# ax.bar(years_of_interest,generation2railway,bottom=consumption2sales+sales2generation,label='Generation→Transportation',color='#808080',alpha=alpha_global)
# ax.bar(years_of_interest,generation2storage,bottom=consumption2sales+sales2generation+generation2railway,label='Generation→Storage',color='green',alpha=alpha_global)
# ax.bar(years_of_interest,generation2PV,bottom=consumption2sales+sales2generation+generation2railway+generation2storage,label='Generation→Generator',color='#C76E00',alpha=alpha_global)


fig,ax=plt.subplots(5,2,width_ratios=[1,1])
ax[0,0].bar(year_list,consumption2sales,color='#237EA8')
ax[1,0].bar(year_list,sales2generation,color='#36AAE2')
ax[2,0].bar(year_list,generation2railway,color='#808080')
ax[3,0].bar(year_list,generation2storage,color='green')
ax[4,0].bar(year_list,generation2PV,color='#C76E00')
for i in range(5):
    ax[i,0].set_ylim((0,0.35))
    ax[i,0].axis('off')

transaction_byprovince_total=pd.DataFrame(index=province_list)
transaction_byprovince_total['consumption2sales']=df_cashflow_consumption2sales.astype(float).mean(axis='columns')/Exchange_rate/1e3
transaction_byprovince_total['sales2generation']=df_cashflow_sales2generation.astype(float).mean(axis='columns')/Exchange_rate/1e3
transaction_byprovince_total['generation2PV']=df_cashflow_generation2PV.astype(float).mean(axis='columns')/Exchange_rate/1e3
transaction_byprovince_total['generation2storage']=df_cashflow_generation2storage.astype(float).mean(axis='columns')/Exchange_rate/1e3
transaction_byprovince_total['generation2railway']=df_cashflow_generation2railway.astype(float).mean(axis='columns')/Exchange_rate/1e3
transaction_byprovince_total['Region']=transaction_byprovince_total.index.map(province_basicinfo['Region'].to_dict())
transaction_byprovince_total_aggregated=transaction_byprovince_total.groupby('Region').agg(sum)

ax[0,1].bar(np.arange(5),transaction_byprovince_total_aggregated['consumption2sales'],color='#237EA8')
ax[1,1].bar(np.arange(5),transaction_byprovince_total_aggregated['sales2generation'],color='#36AAE2')
ax[2,1].bar(np.arange(5),transaction_byprovince_total_aggregated['generation2railway'],color='#808080')
ax[3,1].bar(np.arange(5),transaction_byprovince_total_aggregated['generation2storage'],color='green')
ax[4,1].bar(np.arange(5),transaction_byprovince_total_aggregated['generation2PV'],color='#C76E00')

#%% 安全性的几个指标（对应Fig. 5e）
# 注意要在附录里说明！

# 最困难的部分：供应链安全性
# 在没有标准化前，如果要有零部件缺失，可能要兜回原有省份获取，距离很长（Distance的Weighted Average）；
# 在标准化后，如果要有零部件缺失，直接在本省获取即可
AvgDist_byyear_nonstandard=[]
AvgDist_byyear_standard=[]
china_main_crs_copy=china_main_crs.copy()
china_main_crs_copy['Sample-Points']=china_main_crs_copy['geometry'].sample_points(size=1000)
china_main_crs_copy['Centroid']=china_main_crs_copy['geometry'].centroid
china_main_crs_sample_points=china_main_crs_copy['Sample-Points'].explode(index_parts=True)
df_Avg_distance=pd.Series(0.0,index=province_list)
# 先得到各省与各省中心的距离
for province in province_list:
    #据说EPSG:3857是真实的meter坐标
    sample_points=china_main_crs_sample_points.loc[province].to_crs(3857)
    centroid=gpd.GeoSeries(china_main_crs_copy.loc[province,'Centroid'],index=sample_points.index).set_crs(2343).to_crs(3857)
    df_Avg_distance.loc[province]=sample_points.distance(centroid).mean()/1000

for idx_year,year in enumerate(year_list):
    df_CBS=df_CBStransfer_result_list[idx_year]
    dist_list_standard=[]
    dist_list_nonstandard=[]
    for idx_CBS in df_CBS.index:
        #有一半概率在出发省份搞定，有一半省份在落入省份搞定
        province_from=df_CBS.loc[idx_CBS,'From']
        province_to=df_CBS.loc[idx_CBS,'To']
        dist_list_nonstandard.append((df_Avg_distance[province_from]+df_Avg_distance[province_to]+2*df_CBS.loc[idx_CBS,'Distance'])/4)
        dist_list_standard.append((df_Avg_distance[province_from]+df_Avg_distance[province_to])/2)
    AvgDist_byyear_standard.append(np.average(dist_list_standard,weights=df_CBS['CBSPower']))
    AvgDist_byyear_nonstandard.append(np.average(dist_list_nonstandard,weights=df_CBS['CBSPower']))
print(np.mean(AvgDist_byyear_nonstandard)/np.mean(AvgDist_byyear_standard)) # 2.48倍
# 优势更加体现在在lead time上，可以在正文里说明

# 电网安全性由新能源渗透率决定，可见《环境福祉安全.py》
grid_security_improvement=0.821/0.1658 #(4.95 times)

# 储能柜安全性由液冷效果决定（根据<Effects of different coolants and cooling strategies...>）,
# 可知液冷（油冷）的制冷效果约为空气冷却的1.5-3倍
CBS_security_improvement=(1.5,3)
