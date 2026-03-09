# dm_connect_test.py
import dmPython
import traceback

# 按你的配置填写
HOST = "192.168.112.179"
PORT = 5236
USER = "TEST2026"
PASSWORD = "Dameng123"
SCHEMA = "TEST2026"
try:
    conn = dmPython.connect(
        user=USER,
        password=PASSWORD,
        server=HOST,
        port=PORT,
        schema=SCHEMA,
    )
    cur = conn.cursor()
    cur.execute("SELECT 1")
    print("OK, result:", cur.fetchone())
    cur.close()
    conn.close()
except Exception as e:
    print("CONNECT FAILED:")
    traceback.print_exc()
