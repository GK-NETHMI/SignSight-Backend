import pymysql

try:
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="flask_crud"
    )
    
    print("✅ Database connection successful!")
    conn.close()

except Exception as e:
    print("❌ Database connection failed!")
    print("Error:", e)
