import mariadb
import sys
from jproperties import Properties  # jproperties 라이브러리 사용 => 설치 필요 pip install jproperties
import os

global_conn = None
global_cursor = None

config_path = os.path.join(os.path.dirname(__file__), 'config.properties')

def load_db_config(file_path=config_path):
    configs = Properties()
    with open(file_path, 'rb') as config_file:
        configs.load(config_file)
    
    return {
        'host': configs.get('database.host').data,
        'port': int(configs.get('database.port').data),
        'database': configs.get('database.database').data,
        'user': configs.get('database.user').data,
        'password': configs.get('database.password').data
    }

# def load_db_config(file_path='config.properties'):
#     configs = Properties()
#     with open(file_path, 'rb') as config_file:
#         configs.load(config_file)
    
#     return {
#         'host': configs.get('database.host').data,
#         'port': int(configs.get('database.port').data),
#         'database': configs.get('database.database').data,
#         'user': configs.get('database.user').data,
#         'password': configs.get('database.password').data
#     }

def connect_to_database():
    global global_conn, global_cursor
    try:
        db_config = load_db_config()
        global_conn = mariadb.connect(**db_config)
        global_cursor = global_conn.cursor()
        
        global_cursor.execute("SELECT VERSION()")
        db_version = global_cursor.fetchone()
        print(f"MariaDB 서버에 성공적으로 연결되었습니다. 서버 버전: {db_version[0]}")

        global_cursor.execute("SELECT DATABASE()")
        db_name = global_cursor.fetchone()[0]
        print(f"현재 사용 중인 데이터베이스: {db_name}")

    except mariadb.Error as e:
        print(f"MariaDB에 연결하는 데 실패했습니다: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")
        sys.exit(1)


def close_database_connection():
    global global_conn, global_cursor
    if global_cursor:
        global_cursor.close()
    if global_conn:
        global_conn.close()
        print("MariaDB 연결이 종료되었습니다.")
    
    # 연결을 닫은 후 전역 변수 초기화
    global_conn = None
    global_cursor = None
    

def commit_changes():
    global global_conn
    if global_conn:
        global_conn.commit()
        print("변경사항이 커밋되었습니다.")
    else:
        print("데이터베이스 연결이 없습니다.")

def rollback_changes():
    global global_conn
    if global_conn:
        global_conn.rollback()
        print("변경사항이 롤백되었습니다.")
    else:
        print("데이터베이스 연결이 없습니다.")
