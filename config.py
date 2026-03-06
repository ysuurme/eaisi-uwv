from pathlib import Path

# config.py is in the project root
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Data Directories ---
DIR_DATA_RAW  = PROJECT_ROOT / "data" / "0_raw"
DIR_DB_BRONZE = PROJECT_ROOT / "data" / "1_bronze" / "bronze_data.db"
DIR_DB_SILVER = PROJECT_ROOT / "data" / "2_silver" / "silver_data.db"
DIR_DB_GOLD   = PROJECT_ROOT / "data" / "3_gold"   / "gold_data.db"
DIR_DB_EVAL   = PROJECT_ROOT / "data" / "4_eval" / "eval_data.db"

# --- Toggle Flags ---
START_MLFLOW_UI = True # Set to True/False to auto-start MLflow UI in background

# --- CBS API Configurations ---
# Top 3 most relevant CBS tables for sick leave prediction
CBS_TABLES_T3 = ["80072ned", "83415NED", "83157NED"]

# Top 65 most relevant CBS tables for sick leave prediction
CBS_TABLES_T65 = [
    "80072ned", "83415NED", "86009NED", "86010NED", "85998NED", "86011NED", "86168NED", 
    "83156NED", "84434NED", "83157NED", "84436NED", "83159NED", "84435NED", "84031NED", 
    "84030NED", "83158NED", "85718NED", "85786NED", "86047NED", "86067NED", "81628ENG", 
    "83005ENG", "81174ENG", "81177eng", "81175eng", "85647NED", "80590ned", "85264NED", 
    "85224NED", "85312NED", "83752NED", "85271NED", "83583NED", "83582NED", "85918NED", 
    "85920NED", "81431ned", "83451NED", "85274NED", "85275NED", "85276NED", "85278NED", 
    "81178ENG", "82470NED", "85544NED", "83700ENG", "84432ENG", "71541eng", "82801ENG", 
    "86092NED", "85900NED", "84939NED", "84671NED", "84669NED", "83913NED", "83648NED", 
    "83734NED", "81414NED", "84566NED", "82439NED", "82325NED", "81408NED", "83738NED", 
    "82623NED", "81567NED"
]

# --- Model Storage ---
DIR_MODELS = PROJECT_ROOT / "models"

# --- ML Target Column ---
ML_TARGET_COLUMN = "Ziekteverzuimpercentage_1"


