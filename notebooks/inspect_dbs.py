import sqlite3

# Inspect tasia_db_def1.db
print('=== tasia_db_def1.db ===')
conn = sqlite3.connect(r'c:\Users\jorge\OneDrive\Escritorio\Master CEU\TFM v2\notebooks\tasia_db_def1.db')
cursor = conn.cursor()
cursor.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY type, name")
objects = cursor.fetchall()
for obj in objects:
    print(f'  [{obj[1].upper()}] {obj[0]}')
    cursor.execute(f'SELECT COUNT(*) FROM "{obj[0]}"')
    count = cursor.fetchone()[0]
    cursor.execute(f'PRAGMA table_info("{obj[0]}")')
    cols = cursor.fetchall()
    col_names = [c[1] for c in cols]
    print(f'    Rows: {count}')
    print(f'    Columns: {col_names}')
conn.close()

print()

# Inspect valoralia_db_def1.db
print('=== valoralia_db_def1.db ===')
conn2 = sqlite3.connect(r'c:\Users\jorge\OneDrive\Escritorio\Master CEU\TFM v2\notebooks\valoralia_db_def1.db')
cursor2 = conn2.cursor()
cursor2.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY type, name")
objects2 = cursor2.fetchall()
if not objects2:
    print('  (Base de datos vacía o sin tablas)')
for obj in objects2:
    print(f'  [{obj[1].upper()}] {obj[0]}')
    cursor2.execute(f'SELECT COUNT(*) FROM "{obj[0]}"')
    count2 = cursor2.fetchone()[0]
    cursor2.execute(f'PRAGMA table_info("{obj[0]}")')
    cols2 = cursor2.fetchall()
    col_names2 = [c[1] for c in cols2]
    print(f'    Rows: {count2}')
    print(f'    Columns: {col_names2}')
conn2.close()
