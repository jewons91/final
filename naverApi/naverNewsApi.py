import os
import sys
import urllib.request
import json
import numpy as np
import pandas as pd
client_id = ""
client_secret = ""
encText = urllib.parse.quote("삼성전자")
url = f"https://openapi.naver.com/v1/search/news?query={encText}&display=1&start=1&sort=date" # JSON 결과
# url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # XML 결과
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    response_data = json.loads(response_body.decode('utf-8'))
    
    # json 파일 dump 저장
    # with open('./news_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(response_data, f, indent=4, ensure_ascii=False)
    # print("데이터 저장됨")
    
    # 넘어온 json 파일 어떻게 구성되어있는지 확인
    print(response_body.decode('utf-8'))
    
    # 텍스트 파일 저장
    # with open('./news_data.txt', 'w', encoding='utf-8') as f:
    #     for item in response_data['items']:
    #         f.write(f"Title: {item['title']}\n")
    #         f.write(f"Description: {item['description']}\n")
    #         f.write(f"PublicDate: {item['pubDate']}\n")
    #         f.write("\n")
    
    # DataFrame에 넣어서 csv 파일 저장
    news_items=[]
    for item in response_data['items']:
        news_items.append({
              'Title':item['title']
            , 'Description':item['description']
            , 'Link':item['link']
            , 'PubDate':item['pubDate']
        })
    df = pd.DataFrame(news_items)
    df.to_csv('./news_data.csv', mode='a', header=False, index=False, encoding='utf-8')
    print('csv로 저장됨')
else:
    print("Error Code:" + rescode)
