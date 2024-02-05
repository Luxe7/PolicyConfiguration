
# coding: utf-8

# In[ ]:

tem=DataAPI.IdxGet(secID=u"",indexGroup=u"",consType=u"",consMkt=u"",returnType=u"",
               indexTypeCD=u"",porgFullName=u"",secShortName=u"",wMethodCD="",pubName=u"",industryID=u"",sortID=u"",field=u"",pandas="1")


# In[ ]:

tem


# In[ ]:

tem.info()


# In[ ]:




# In[ ]:

encoded_str = '\xe7\xa7\x81\xe5\x8b\x9f\xe7\xbb\xbc\xe5\x90\x88\xe6\x8c\x87\xe6\x95\xb0\xef\xbc\x88\xe9\x80\x9a\xe8\x81\x94\xef\xbc\x89'
decoded_str = encoded_str.decode('utf-8')

print(decoded_str)


# In[ ]:

unique_values = tem['indexType'].unique()
print(unique_values[1])
for i in unique_values:
    print(i)
# 使用encode方法将字符串转换为指定的字符编码，例如UTF-8
unique_values_str = [value.decode('utf-8') for value in unique_values]
unique_values_str


# In[ ]:

tem[tem['indexType']=='策略指数']


# In[ ]:

tem[(tem['indexType']=='策略指数') & (tem['ticker'].str.contains('930'))].columns


# In[ ]:

df=DataAPI.MktIdxdGet(indexID=u"",ticker=u"",tradeDate=u"20161219",beginDate=u"",endDate=u"",exchangeCD=u"XSHE,XSHG",field=u"",pandas="1")


# In[ ]:

df.head()


# In[ ]:

df[df['indexID']=='000138.ZICN']


# In[ ]:

import pandas as pd

# 生成日期范围，从2016年1月1日到2018年12月31日
date_range = pd.date_range(start='2007-01-04', end='2023-08-30')

# 将日期格式化为字符串
formatted_dates = date_range.strftime('%Y-%m-%d')

# 打印结果
print(formatted_dates)
len(formatted_dates)


# In[ ]:

numbers_list = [
    399372, 399373, 399374, 399375, 399376, 399377,
    '000925', '000965', '000966', '000967', '000052', '000053', '000128', '930723',
    'H11110', '000015', 'H30094', '000825', '000129', '000803', '000130', '000804', '930846',
    '000135', '000136', '000137', '000138', '000828', '000829', '000830', '000831', 'H50045',
    'H50046', 'H50047', 'H50057', 'H50048', 'H50061', 'H50062', 'H50063', 'H50048', 'H50050',
    'H30347', '932053', '931588', '931155', '930667', 'H30269', 'H30270', '931468', '931056',
    '950179', '950095', '950094', 'H50066', '930044', 'H30322', '930929', '930666', '931259',
    '931129', '930860', '930949', '931133', '931498', '931067', '931476', '931697'
]

# 使用列表推导式将所有元素转换为字符串类型
str_numbers_list = [str(item) for item in numbers_list]

print(str_numbers_list)
len(str_numbers_list),sorted(str_numbers_list)


# In[ ]:

df=DataAPI.MktIdxdGet(indexID=u"",ticker=str_numbers_list,tradeDate=u"20161219",beginDate=u"",endDate=u"",exchangeCD=u"",field=u"",pandas="1")


# In[ ]:

df


# In[ ]:

select_data=df['ticker'].tolist()


# In[ ]:

import pandas as pd

data = {'Group': ['A', 'B', 'A', 'B', 'A', 'B'],
        'Value': [10, 20, 30, 40, 50, 60]}

df = pd.DataFrame(data)
print("原始 DataFrame:")
print(df)
# 按照 'Group' 列进行分组
grouped_df = df.groupby('Group')

# 打印分组后的结果
for group_name, group_data in grouped_df:
    print(group_name)
    print(group_data)


# ## 获取数据

# In[ ]:

df=DataAPI.MktIdxdGet(indexID=u"",ticker=str_numbers_list,tradeDate=u"",beginDate=u"20070104",endDate=u"20230830",exchangeCD=u"",field=u"",pandas="1")


# In[ ]:

df


# In[ ]:

df=df[['ticker','secShortName','tradeDate','closeIndex']]
df['ticker']=df['ticker'].astype(str)
df[['ticker','closeIndex']]


# In[ ]:


df.to_csv("close_index.csv",index=False)


# In[ ]:

true_date=df[df['ticker']=='000015']['tradeDate'].tolist()
df_left = pd.DataFrame({'date':true_date})
len(true_date),df_left.head()


# In[ ]:

dict18 = {
    'SZ50':'000016','SZ180':'000010','SZ380':'000009','SZZZ':'000001','SSAG':'000002','SSBG':'000003','SHZAZ':'399107','SHZBZ':'399108','SHZZZ':'399106','SHZCZ':'399001','HS300':'399300','ZZ100':'000903','ZZ500':'399905','ZZ800':'000906','ZZ1000':'000852','ZX300':'399008','ZXBZHI':'399005','ZXBZ':'399101'}
list18 = dict18.values()
for id in list18:
    iID =id+'.ZICN'
    saveFile = './day/DAY_'+id+'.csv'
    df_day = DataAPI.MktIdxdGet(indexID=iID,ticker=u"",tradeDate=u"",beginDate=u"",endDate=u"20190625",exchangeCD=u"",field=u"indexID,secShortName,tradeDate,preCloseIndex,closeIndex",pandas="1")
    df_day.to_csv(saveFile,index=False)
    # keyn =  new_dict[id]
    # print(keyn,id)
    print(df_day.head(1))  


# In[ ]:

# 按照 'Group' 列进行分组
grouped_df = df.groupby('ticker')


# 打印分组后的结果
for group_name, group_data in grouped_df:
    file_name=group_name+'_'+group_data['secShortName'].iloc[0]+".csv"
    result_df = pd.merge(df_left, group_data, left_on='date', right_on='tradeDate', how='left')[['date','closeIndex']] #为了确保日期是正确的
    result_df= result_df.set_index('date').T
    print(group_name,group_data.shape[0],file_name)
    # print(result_df)
    result_df.to_csv('./close_index/'+file_name,index=False)
    



# In[ ]:

df=pd.read_csv("close_index.csv")


# In[ ]:

df


# In[ ]:



