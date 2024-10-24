import requests
import json
import numpy as np
import pandas as pd

APP_KEY = ""
APP_SECRET = ""
ACCESS_TOKEN = ""
URL_BASE = "https://openapi.koreainvestment.com:9443" #실전투자

# 주식현재가 시세
def get_current_price(stock_no):
    PATH = "uapi/domestic-stock/v1/quotations/news-title"
    URL = f"{URL_BASE}/{PATH}"

    # 헤더 설정
    headers = {"Content-Type":"application/json", 
            "authorization": f"Bearer {ACCESS_TOKEN}",
            "appKey":APP_KEY,
            "appSecret":APP_SECRET,
            "tr_id":"FHKST01011800",
            "custtype": 'P'}

    params = {
        "FID_NEWS_OFER_ENTP_CODE": ""
        , "FID_COND_MRKT_CLS_CODE": ""
        , "FID_INPUT_ISCD": stock_no
        , "FID_TITL_CNTT": ""
        , "FID_INPUT_DATE_1": "0020240616"
        , "FID_INPUT_HOUR_1": ""
        , "FID_RANK_SORT_CLS_CODE": ""
        , "FID_INPUT_SRNO": ""
    }

    # 호출
    res = requests.get(URL, headers=headers, params=params)
    print(ACCESS_TOKEN)
    if res.status_code == 200 and res.json()["rt_cd"] == "0" :
        return(res.json())
    else:
        print("Error Code : " + str(res.status_code) + " | " + res.text)
        return None

result = get_current_price("003230")
print(type(result))
df = pd.DataFrame(result['output'])
# df.rename(columns={
#       'stck_bsop_date': '주식 영업 일자'
#     , 'stck_oprc': '주식 시가'
#     , 'stck_hgpr': '주식 최고가'
#     , 'stck_lwpr': '주식 최저가'
#     , 'stck_clpr': '주식 종가'
#     , 'acml_vol': '누적 거래량'
#     , 'prdy_vrss_vol_rate': '전일 대비 거래량 비율'
#     , 'prdy_vrss': '전일 대비'
#     , 'prdy_vrss_sign': '전일 대비 부호'
#     , 'prdy_ctrt': '전월 대비율'
#     , 'hts_frgn_ehrt': 'HTS 외국인 소진율'
#     , 'frgn_ntby_qty': '외국인 순매수 수량'
#     , 'flng_cls_code': '락 구분 코드'
#     , 'acml_prtt_rate': '누적 분할 비율'
# }
        #   , inplace=True)
with pd.ExcelWriter('./news_data.xlsx') as w:
    df.to_excel(w, sheet_name='TEST')
