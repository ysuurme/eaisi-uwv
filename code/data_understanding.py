import cbsodata
import sqlite3

# The 'con' object represents the connection to the on-disk database.
con = sqlite3.connect("data//1_bronze//bronze_data.db")

# the database cursor 'cur' object is required to execute and fetch results from SQL queries
cur = con.cursor()

cur.execute("CREATE TABLE sample_org(name, alias, superpower)")

# Verify new table creation by querying the sqlite_master table built-in
result = cur.execute("SELECT name FROM sqlite_master")
result.fetchone()

# data = cbsodata.get_data('82070ENG', dir="data\\0_raw")