# import pymysql

# def connection():
#     connection = pymysql.connect(
#         host='192.168.40.53',       # 데이터베이스 호스트
#         user='root',                # 사용자 이름
#         password='big185678',       # 비밀번호
#         database='finaldb',         # 데이터베이스 이름
#     )
#     return connection

def stock_shares_dict():
    stock_shares_dict = {
        'SK하이닉스': 728002365,
        '포스코퓨처엠': 77463220,
        '현대차': 209416191,
        'POSCO홀딩스': 82624377,
        '삼성전자': 5969782550,
        '포스코DX': 152034729,
        '한미반도체': 96993634,
        '에코프로': 133138340,
        '알테오젠': 53148528,
        '에코프로비엠': 97801344
    }
    return stock_shares_dict

def stock_dict():
    stock_dict = {
        'SK하이닉스': '000660',
        '포스코퓨처엠': '003670',
        '현대차': '005380',
        'POSCO홀딩스': '005490',
        '삼성전자': '005930',
        '포스코DX': '022100',
        '한미반도체': '042700',
        '에코프로': '086520',
        '알테오젠': '196170',
        '에코프로비엠': '247540'
    }
    return stock_dict

def stock_code():
    stock_code = ['000660','003670','005380','005490','005930','022100','042700','086520','196170','247540']
    return stock_code

def stock_shares():
    stock_shares = {
            '728002365',
            '77463220',
            '209416191',
            '82624377',
            '5969782550',
            '152034729',
            '96993634',
            '133138340',
            '53148528',
            '97801344'}
    return stock_shares

def select_sql(code):
    query = f"""
            SELECT am.code
                , am.close_price 
                , am.volume 
              FROM A{code}_mindata am 
            ;
            """
    return query

def execute_query(code):
    conn = connection()
    if conn is None:
        return None
    
    try:
        query = select_sql(code)
        with conn.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()  # 모든 결과를 가져옵니다.
            return result
    except pymysql.MySQLError as e:
        print(f"Error executing query: {e}")
        return None
    finally:
        conn.close()  # 커넥션 종료



def column_rename(df):
    df.rename(columns={
                    0 : 'code',
                    1 : '종가',
                    2 : '거래량',
                    }, inplace=True)

