import os
import psycopg2
from dotenv import load_dotenv

load_dotenv('services/api/.env')
db_url = os.environ.get('DATABASE_URL')
conn = psycopg2.connect(db_url)
cur = conn.cursor()

cur.execute("""
    SELECT column_name, data_type, udt_name 
    FROM information_schema.columns 
    WHERE table_name = 'incidents'
    ORDER BY ordinal_position;
""")
print("COLUMNS:")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]} ({row[2]})")

cur.execute("""
    SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'incidents';
""")
print("\nINDEXES:")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]}")

conn.close()

