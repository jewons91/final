import os
import sys
import urllib.request
import json
import numpy as np
import pandas as pd
client_id = "u3k3cwi0Us5G4eEihXAK"
client_secret = "40G6XWbyoy"
encText = urllib.parse.quote("짜장면")
url = f"https://openapi.naver.com/v1/search/blog?query={encText}&display=1&start=1&sort=date" # JSON 결과
# url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # XML 결과
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    response_data = json.loads(response_body.decode('utf-8'))
    
    with open('./blog_data.json', 'w', encoding='utf-8') as f:
        json.dump(response_data, f, indent=4, ensure_ascii=False)
    print("데이터 저장됨")
    print(response_body.decode('utf-8'))
    
    with open('./blog_data.txt', 'w', encoding='utf-8') as f:
        for item in response_data['items']:
            f.write(f"Title: {item['title']}\n")
            f.write(f"Description: {item['description']}\n")
            f.write(f"PostDate: {item['postdate']}\n")
            f.write("\n")
            
    blog_items=[]
    for item in response_data['items']:
        blog_items.append({
              'Title':item['title']
            , 'Description':item['description']
            , 'PostDate':item['postdate']
        })
    df = pd.DataFrame(blog_items)
    df.to_csv('./blog_data.csv', mode='w', header=False, index=False, encoding='utf-8')
    print('csv로 저장됨')
else:
    print("Error Code:" + rescode)