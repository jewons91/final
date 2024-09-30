import pandas as pd

def csvDataLoad(code):
    data = pd.read_csv(f'./{code}.csv',encoding='cp949')
    data = data[['code','종가', '거래량']]
    return data

def fdata():
    data=pd.read_csv('C:/Users/F06/Downloads/fdata.csv')
    return data

def adlldata():
    data = fdata()
    data = dataRenameVol(data)
    return data
    

def dataLoad(code):
    data = pd.read_csv('C:/Users/F06/Downloads/fdata.csv')
    data = data[data['code']==code]
    data = dataRename(data)
    return data

def dataRename(data):
    data=data[['종가','매도량','매수량']]
    data.rename(columns={
                                '종가' : 'close',
                                '매도량' : 'sell_volume',
                                '매수량' : 'buy_volume',
                            }, inplace=True)
    return data

def dataRenameVol(data):
    data=data[['종가', '거래량']]
    data.rename(columns={
                                '종가' : 'close',
                                '거래량' : 'volume',
                            }, inplace=True)
    return data

def data_000660():
    data = pd.read_csv('./000660.csv')
    data = dataRenameVol(data)
    return data

def dataRename2(data):
    data = pd.read_csv('./000660.csv')
    data=data[['날짜','시간','종가']]
    data[['날짜','시간']] = data[['날짜', '시간']].astype(str)
    data['date'] = data['날짜'].astype(str) + data['시간'].astype(str)
    data['시간'] = data['시간'].str.zfill(4)
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d%H%M')
    data = data.drop(['날짜'],axis=1)
    data = data.drop(['시간'],axis=1)
    data.rename(columns={'종가':'SP500','date':'ds'}, inplace=True)
    return data
    