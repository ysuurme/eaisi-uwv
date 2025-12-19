
import sqlite3

"""
Database: https://docs.python.org/3/library/sqlite3.html
Pipelines: sqlalchemy, pdantic
"""

# The 'con' object represents the connection to the on-disk database.
con = sqlite3.connect("data//1_bronze//bronze_data.db")

# the database cursor 'cur' object is required to execute and fetch results from SQL queries
cur = con.cursor()

cur.execute("CREATE TABLE sample_org(name, alias, superpower)")

# Verify new table creation by querying the sqlite_master table built-in
result = cur.execute("SELECT name FROM sqlite_master")
result.fetchone()

data = [
    ("Sepehr Harsiny", "The Illuminator", "Insight Extraction (Uncovering the crucial findings from data and research)"),
    ("Dennis Snijders", "The Polymath", "Adaptive Versatility (Excelling in any task: modeling, data engineering, or project communication)"),
    ("Yannic Suurmeijer", "The Architecht", "Code Synthesis (Building robust, scalable code and infrastructure)"),
    ("Ruud van Cruchten", "The Navigator", "Structural Integrity (Ensuring the project stays on course and roles are clear)"),
]

cur.executemany("INSERT INTO sample_org VALUES(?, ?, ?)", data)
con.commit()  # Remember to commit the transaction after executing INSERT

for row in cur.execute("SELECT * FROM sample_org"):
    print(row)

con.close()

# data = cbsodata.get_data('82070ENG', dir="data\\0_raw")