import pandas as pd
import sys
import os

# 현재 파일의 디렉토리 경로를 가져와서 final 디렉토리 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from DB import conn

# 주식 종목 번호
stock_no_list = ['005930', '000660', '042700', '030200', '017670', '097950', '004370', '003230']

for stock_no in stock_no_list:
    # 과거 데이터 데이터프레임
    past_mins_df = pd.read_csv(f'hangukInvestApi/past_mins/stock_price-{stock_no}.csv')

    # 데이터가 int64로 이루어져 있어서 str 타입으로 변환
    past_mins_df = past_mins_df.astype(str)

    # stck_bsop_date를 날짜와 시간으로 분리
    past_mins_df['stck_cntg_hour'] = past_mins_df['stck_bsop_date'].str[8:].str.pad(width=6, side='right', fillchar='0')
    past_mins_df['stck_bsop_date'] = past_mins_df['stck_bsop_date'].str[:8]

    # insertdate 생성
    past_mins_df['insertdate'] = \
        (pd.to_datetime(past_mins_df['stck_bsop_date'], format='%Y%m%d') + pd.DateOffset(days=1)).dt.strftime('%Y%m%d')

    # DB 연결
    conn.connect_to_database()

    # GSTC_CODE 및 INVEST_CODE 가져오기
    code_select_query = f'''
        SELECT a.GSTC_CODE, a.INVEST_CODE
        FROM TB_STOCKCLASSIFY a
        WHERE a.KSTC_CODE = {stock_no}
        '''

    try:
        conn.global_cursor.execute(code_select_query)
        # 결과를 데이터프레임으로 반환
        columns = [column[0] for column in conn.global_cursor.description]
        data = conn.global_cursor.fetchall()
        select_df = pd.DataFrame(data, columns=columns)
    except Exception as e:
        print(f'SELECT 중 오류 발생: {e}')

    # 데이터프레임에 CODE 추가
    for column in select_df.columns:
        past_mins_df[column] = select_df[column].iloc[0]
        print(past_mins_df)

    # 데이터프레임에서 튜플 리스트 생|성
    data_tuples = [tuple(x) for x in past_mins_df.to_numpy()]

    # TB_MINSSTOCK 테이블에 넣을 쿼리문 작성
    insert_query = '''
        INSERT INTO TB_MINSSTOCK
        (stck_bsop_date, stck_oprc, stck_hgpr, stck_lwpr, stck_prpr, cntg_vol, stck_cntg_hour, insertdate, gstc_code, invest_code)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''

    # DB에 데이터 삽입
    try:
        conn.global_cursor.executemany(insert_query, data_tuples)
        conn.commit_changes()
        print('데이터 적재 완료')
    except conn.mariadb.Error as e:
        print(f'데이터 적재 중 오류 발셍: {e}')
        conn.rollback_changes()

    # DB 연결 닫기
    conn.close_database_connection()

print('과거 데이터 적재 완료')