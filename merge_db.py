import sqlite3
import os

def merge_sqlite_dbs(source_db1, source_db2, destination_db):
    # Copy source_db1 as the base of destination_db
    if os.path.exists(destination_db):
        os.remove(destination_db)
    with open(source_db1, 'rb') as fsrc, open(destination_db, 'wb') as fdst:
        fdst.write(fsrc.read())

    # Connect to destination and source2
    dest_conn = sqlite3.connect(destination_db)
    src2_conn = sqlite3.connect(source_db2)
    dest_cur = dest_conn.cursor()
    src2_cur = src2_conn.cursor()

    # Get list of tables
    src2_cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in src2_cur.fetchall()]

    for table in tables:
        print(f"Merging table: {table}")

        # Get column names
        src2_cur.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in src2_cur.fetchall()]
        col_str = ', '.join(columns)
        placeholder_str = ', '.join('?' * len(columns))

        # Read data from source2
        src2_cur.execute(f"SELECT {col_str} FROM {table}")
        rows = src2_cur.fetchall()

        # Insert into destination using INSERT OR IGNORE to skip duplicates
        dest_cur.executemany(
            f"INSERT OR IGNORE INTO {table} ({col_str}) VALUES ({placeholder_str})", rows
        )

    dest_conn.commit()
    dest_conn.close()
    src2_conn.close()
    print(f"Merge completed into: {destination_db}")

# Example usage
merge_sqlite_dbs('words/data1.db', 'words/data2.db', 'words/merged.db')

