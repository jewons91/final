import inquire_daily_itemchartprice1 as idi
import pandas as pd
from datetime import datetime
import calendar
from dateutil.relativedelta import relativedelta
import time

import sys
import os
# 현재 파일의 디렉토리 경로를 가져와서 final 디렉토리 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# 이제 conn 모듈을 임포트할 수 있습니다.
from DB import conn

def fetch_daily_price(col):
    stock_no = col # 주식 종목 번호
    year = 2000 # 시작 연도
    month = 1 # 시작 월

    # 현재 날짜
    current_date = datetime.now()

    # 시작 날짜
    date = datetime(year, month, 1)

    result_df = pd.DataFrame()
    
    # GSTC_CODE 및 INVEST_CODE 가져올 SELECT문 작성
    select_query = f'''
        SELECT a.GSTC_CODE, a.INVEST_CODE
        FROM TB_STOCKCLASSIFY a
        WHERE a.KSTC_CODE = '{stock_no}'
        '''
    # DB에서 select해 오고 데이터프레임에 넣기
    try:
        conn.global_cursor.execute(select_query)
        # 결과를 데이터프레임으로 변환
        columns = [column[0] for column in conn.global_cursor.description]
        data = conn.global_cursor.fetchall()
        select_df = pd.DataFrame(data, columns=columns)
    except Exception as e:
        print(f'오류 발생: {e}')
        
    # 적재되어 있는 가장 최근 날짜 확인
    select_date_check_query = f'''
        SELECT a.STCK_BSOP_DATE
          FROM TB_DAILYSTOCK a
         WHERE a.GSTC_CODE = '{select_df.iloc[0, 0]}'
         ORDER BY a.STCK_BSOP_DATE DESC
        '''
    # DB에서 select해 오고 데이터프레임에 넣기
    try:
        conn.global_cursor.execute(select_date_check_query)
        # 결과를 데이터프레임으로 변환
        columns = [column[0] for column in conn.global_cursor.description]
        data = conn.global_cursor.fetchall()
        select_date_check_df = pd.DataFrame(data, columns=columns)
    except Exception as e:
        print(f'오류 발생: {e}')
    
    if select_date_check_df['STCK_BSOP_DATE'].notnull().any():
        last_date = datetime.strptime(select_date_check_df.iloc[0, 0], '%Y%m%d')
        date = last_date + relativedelta(days=1)
        
    # 일일 데이터 가져오기
    while date <= current_date:
        # 해당 달의 첫째 날짜
        first_day = date
        # 해당 달의 마지막 날짜
        last_day = datetime(date.year, date.month, calendar.monthrange(date.year, date.month)[1])
        
        if date + relativedelta(months=1) > current_date:
            last_day = current_date
        
        # datetime을 string으로 변환
        first_string = first_day.strftime('%Y%m%d')
        last_string = last_day.strftime('%Y%m%d')
        
        result = idi.get_daily_price(stock_no, first_string, last_string)
        time.sleep(0.05)
        
        if result['output2'][0]:
            df = pd.DataFrame(result['output2'])
            # df['insertdate'] = pd.to_datetime(df['stck_bsop_date']) + pd.Timedelta(days=1)
            df['insertdate'] = datetime.now()
            df = df.dropna()
            result_df = pd.concat([result_df, df], ignore_index=True)
        else:
            print('NULL')
            # break
        date += relativedelta(months=1)
    
    ntn_insert_df = None
    if result_df is not None and not result_df.empty:
        # result_df select_df 합치기
        expanded_select_df = pd.concat([select_df] * len(result_df), ignore_index=True)
        insert_df = pd.concat([expanded_select_df.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)

        # insert_df의 Nan을 None으로 변환
        ntn_insert_df = insert_df.where(pd.notnull(insert_df), '')
        # 'insertdate'를 문자열 형식으로 변환(적재 위해서)
        # ntn_insert_df['insertdate'] = ntn_insert_df['insertdate'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    if ntn_insert_df is not None and not ntn_insert_df.empty:
        json_data = ntn_insert_df.to_json(orient='records', date_format='iso')
        print(json_data)
        return ntn_insert_df, json_data
    else:
        print('No Data')
        return None, None

def main():
    # DB 연결
    conn.connect_to_database()

    # 주식 종목 번호 리스트로 가져오기
    all_code_select_query = '''
        SELECT a.KSTC_CODE
        FROM TB_STOCKCLASSIFY a
        '''
    
    # 가져오기
    try:
        conn.global_cursor.execute(all_code_select_query)
        # 결과를 데이터프레임으로 반환
        columns = [column[0] for column in conn.global_cursor.description]
        data = conn.global_cursor.fetchall()
        all_select_df = pd.DataFrame(data, columns=columns)
    except Exception as e:
        print(f'SELECT 중 오류 발생: {e}')
    stock_no_list = all_select_df['KSTC_CODE'].values.tolist()

    # stock_no_list = ['005930', '000660', '042700', '030200', '017670', '097950', '004370', '003230']
    for col in stock_no_list:
        ntn_insert_df, _ = fetch_daily_price(col)
        
        if ntn_insert_df is not None and not ntn_insert_df.empty:
            # 데이터프레임에서 튜플 리스트 생성
            data_tuples = [tuple(x) for x in ntn_insert_df.to_numpy()]

            # TB_DAILYSTOCK 테이블에 넣을 쿼리문 작성
            insert_query = '''
                INSERT INTO TB_DAILYSTOCK 
                (GSTC_CODE, INVEST_CODE, STCK_BSOP_DATE, STCK_CLPR, STCK_OPRC, STCK_HGPR, STCK_LWPR, ACML_VOL, ACML_TR_PBMN, FLNG_CLS_CODE, PRTT_RATE, MOD_YN, PRDY_VRSS_SIGN, PRDY_VRSS, REVL_ISSU_REAS, INSERTDATE)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                '''

            # DB에 데이터 삽입
            try:
                conn.global_cursor.executemany(insert_query, data_tuples)
                conn.commit_changes()
                print('데이터 적재 완료')
            except Exception as e:
                print(f'데이터 적재 중 오류 발생: {e}')
                conn.rollback_changes()

    # DB 연결 닫기
    conn.close_database_connection()
    
if __name__ == '__main__':
    main()