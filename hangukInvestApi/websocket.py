# 웹 소켓 모듈을 선언한다.
import websockets
import json
import requests
import os
import asyncio
import time

clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')

key_bytes = 32

# AES256 DECODE
# def aes_cbc_base64_dec(key, iv, cipher_text):
#     """
#     :param key:  str type AES256 secret key value
#     :param iv: str type AES256 Initialize Vector
#     :param cipher_text: Base64 encoded AES256 str
#     :return: Base64-AES256 decodec str
#     """
#     cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))
#     return bytes.decode(unpad(cipher.decrypt(b64decode(cipher_text)), AES.block_size))

# 웹소켓 접속키 발급
def get_approval(key, secret):
    # url = https://openapivts.koreainvestment.com:29443' # 모의투자계좌     
    url = 'https://openapi.koreainvestment.com:9443' # 실전투자계좌
    headers = {"content-type": "application/json"}
    body = {"grant_type": "client_credentials",
            "appkey": key,
            "secretkey": secret}
    PATH = "oauth2/Approval"
    URL = f"{url}/{PATH}"
    res = requests.post(URL, headers=headers, data=json.dumps(body))
    approval_key = res.json()["approval_key"]
    return approval_key

async def connect():
    # 웹 소켓에 접속.( 주석은 koreainvest test server for websocket)
    ## 시세데이터를 받기위한 데이터를 미리 할당해서 사용한다.
    
    g_appkey = 'PSrbq88Esne4tZ3xDS3PutsAHxAgXwZvLXtf'
    g_appsceret = 'AnjXzmz+xz2RX+af/HJ6O5OmOknDW/FcEOcya1Xd53xppa/X303L2P4u+s3CgviQ+oImayLhkELTHo2FcNbmuxARgCqBnoxiU47Nut4qDfDUeH1Ps0fnP9g+thecbadLB7bNHgY55mkoLxLSsjudlmnXv/4s4Pnpij7CfB28ZbasD9yQuNI='
    g_token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0b2tlbiIsImF1ZCI6IjkyMzVkYjg5LTFhZDQtNGZiNy05OGViLWY5MDk4MDRiOWNmOSIsInByZHRfY2QiOiIiLCJpc3MiOiJ1bm9ndyIsImV4cCI6MTcyMTM3ODEzNCwiaWF0IjoxNzIxMjkxNzM0LCJqdGkiOiJQU3JicTg4RXNuZTR0WjN4RFMzUHV0c0FIeEFnWHdadkxYdGYifQ.zjyALaUxj7_QtwCTakhfFQw08RplZ-XoaD_4ALkBe4aufYAFeCMsj_IYjXi_5l7cwQ3Un7FHd0IItj9mfN0dMw'
    
    stockcode = '005930'    # 테스트용 임시 종목 설정, 삼성전자
    # htsid = 'HTS ID를 입력하세요'    # 체결통보용 htsid 입력
    # custtype = 'P'      # customer type, 개인:'P' 법인 'B'
    
    # url = 'ws://ops.koreainvestment.com:31000' # 모의투자계좌
    # url = 'ws://ops.koreainvestment.com:21000' # 실전투자계좌
    url = 'wss://openapi.koreainvestment.com:9443'
    
    # g_approval_key = get_approval(g_appkey, g_appsceret)
    g_approval_key = 'fe26dac2-597e-42ad-8f7b-47f861eb1a55'
    print("approval_key [%s]" % (g_approval_key))
    
    async with websockets.connect(url, ping_interval=None) as websocket:

        # senddata = f'{"header":{"approval_key":"{g_approval_key}", "content-type":"application/json", "authorization":"Bearer {g_token}", "tr_id":"FHKST03010200", "custtype":"'+custtype+'","tr_type":"' + tr_type + '","content-type":"utf-8"},"body":{"input":{"tr_id":"' + tr_id + '","tr_key":"' + htsid + '"}}}'
        senddata = f'{{"header":{{"approval_key":"{g_approval_key}", "content-type":"application/json", "authorization":"Bearer {g_token}", "tr_id":"FHKST03010200", "custtype":"P"}}, "body":{{"FID_ETC_CLS_CODE":"", "FID_COND_MRKT_DIV_CODE":"J", "FID_INPUT_ISCD":"{stockcode}", "FID_INPUT_HOUR_1":"", "FID_PW_DATA_INCU_YN":"Y"}}}}'
        
        print('Input Command is :', senddata)

        await websocket.send(senddata)
        # 무한히 데이터가 오기만 기다린다.
        while True:
            data = await websocket.recv()
            print("Recev Command is :", data)
            if data['rt_cd'] == '0' or data['rt_cd'] == '1':  # 실시간 데이터일 경우
                jsonObject = json.load(data)
                trid = jsonObject["header"]["tr_id"]

                
                if data['rt_cd'] == '0':
                    output2 = jsonObject['output2']
                    print(output2)
                    await asyncio.sleep(1)


# 비동기로 서버에 접속한다.
asyncio.get_event_loop().run_until_complete(connect())
asyncio.get_event_loop().close()