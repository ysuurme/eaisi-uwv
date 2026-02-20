import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, select, MetaData
from pathlib import Path

# Path to bronze DB
bronze_db_path = Path("data/1_bronze/bronze_data.db")
engine = create_engine(f"sqlite:///{bronze_db_path}")
metadata = MetaData()
metadata.reflect(bind=engine)

# Get tables for 80072ned
fact_table = metadata.tables["80072ned_fact"]
dim_bedrijf = metadata.tables["80072ned_dim_BedrijfskenmerkenSBI2008"]
dim_perioden = metadata.tables["80072ned_dim_Perioden"]

# Build query: Join fact with dimensions
query = select(
    fact_table.c.ID,
    fact_table.c.BedrijfskenmerkenSBI2008,
    fact_table.c.Perioden,
    fact_table.c.Ziekteverzuimpercentage_1,
    dim_bedrijf.c.Title.label("Bedrijfskenmerken_Title"),
    dim_perioden.c.Title.label("Perioden_Title")
).join(
    dim_bedrijf, fact_table.c.BedrijfskenmerkenSBI2008 == dim_bedrijf.c.Key, isouter=True
).join(
    dim_perioden, fact_table.c.Perioden == dim_perioden.c.Key, isouter=True
)

# Execute and load into DataFrame
with engine.connect() as conn:
    df = pd.read_sql(query, conn)

print(f"Loaded {len(df)} rows from the joined table.")
print(df.head())

# Basic visualization: Bar chart of average Ziekteverzuimpercentage_1 by Bedrijfskenmerken_Title (top 10)
if not df.empty:
    print("Data loaded successfully. Creating visualization...")
    # Group by category and calculate mean
    grouped = df.groupby("Bedrijfskenmerken_Title")["Ziekteverzuimpercentage_1"].mean().sort_values(ascending=False).head(10)

    # Plot
    plt.figure(figsize=(10, 6))
    grouped.plot(kind='bar')
    plt.title("Average Ziekteverzuimpercentage by Bedrijfskenmerken (Top 10)")
    plt.xlabel("Bedrijfskenmerken Title")
    plt.ylabel("Average Ziekteverzuimpercentage")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("bronze_visualization.png")
    print("Visualization saved to bronze_visualization.png")
else:
    print("No data found in the joined table.")