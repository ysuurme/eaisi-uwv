from pathlib import Path

# --- Data Directories ---
DIR_DATA_RAW = Path("data/0_raw")
DIR_DB_BRONZE = Path("data/1_bronze/bronze_data.db")
DIR_DB_SILVER = Path("data/2_silver/silver_data.db")
DIR_DB_GOLD = Path("data/3_gold/gold_data.db")

# --- CBS API Configurations ---
# Top 3 most relevant CBS tables for sick leave prediction
# CBS_TABLES_T3 = ["80072ned", "83415NED", "83157NED"]

# Top 65 most relevant CBS tables for sick leave prediction
CBS_TABLES_T65 = [
    ## Direct Sick Leave & Absence Data
    "80072ned", # Sick leave percentage by industry and size
    "83415NED", # Sick leave by working conditions
    # "86009NED", # Sick leave by industry and branch size
    # "86010NED", # Sick leave by occupation/profession
    # "85998NED", # Sick leave by gender and age                                                    ---------> Gender and age are sensitive topics let's skip it for the beginning
    # "86011NED", # Sick leave by origin and education level
    # "86168NED", # Absence duration by type of complaint/reason
    
    ## Workplace Surveys & Working Conditions
    # "83156NED", # Sustainable employability by industry
    # "84434NED", # Sustainable employability by occupation
    # "83157NED", # Psychosocial work load (PSA) by industry
    # "84436NED", # Psychosocial work load (PSA) by occupation
    # "83159NED", # Physical work load by industry
    # "84435NED", # Physical work load by occupation
    # "84031NED", # Occupational accidents (4+ days absence) by company characteristics
    # "84030NED", # Occupational accidents by accident characteristics
    # "83158NED", # Occupational accidents by industry
    # "85718NED", # Work from home (Company Policy proxy)
    # "85786NED", # Weekly work-from-home hours
    
    ## General Health & Lifestyle (Predictive Proxies)
    # "86047NED", # Labor participation and health status
    # "86067NED", # Health status of the labor force
    # "81628ENG", # Health, lifestyle, and healthcare supply key figures
    # "83005ENG", # Perceived state of health and medical care contacts
    # "81174ENG", # Chronic disorders and functional limitations
    # "81177eng", # Lifestyle and preventive screening
    # "81175eng", # Lifestyle factors by sex and age
    # "85647NED", # Health and labor barriers for the non-working population
    
    ## Labor Market & Unemployment (Economic Pressure)
    # "80590ned", # Labor participation and unemployment per month
    # "85264NED", # General labor participation core figures
    # "85224NED", # Labor participation (seasonally adjusted)
    # "85312NED", # Unemployment duration (long-term unemployment trends)
    # "83752NED", # Long-term labor participation trends (since 1969)
    # "85271NED", # Average income and labor position
    
    ## Industry-Specific Trends & Workforce Characteristics
    # "83583NED", # Jobs by industry (SBI 2008) and size
    # "83582NED", # Jobs by industry and region
    # "85917NED", # Labor volume by industry and age
    # "85918NED", # Labor volume by industry and gender
    # "85920NED", # Labor volume by industry (quarterly)
    # "85921NED", # Labor volume by industry and region                                               -----------> Not downloaded due to a 404 error.
    # "85922NED", # Labor volume by industry and size
    # "81431NED", # Employment, wages, and working hours (core figures)
    # "83451NED", # Monthly employment and wage trends
    # "85274NED", # Seniority (tenure) of the working population
    # "85275NED", # Average working hours
    # "85276NED", # Occupational distribution
    # "85278NED", # Position in the work sphere (Fixed vs. Flexible contracts)
    
    ## Healthcare Accessibility & Costs
    # "81178ENG", # Medical contacts, hospitalization, and medicine use
    # "82470NED", # Medical specialist care by diagnosis (DBCs)
    # "85544NED", # Healthcare expenditure by financing and function
    # "83700ENG", # Revenues of health care financing schemes
    
    ## Macroeconomic Proxies (GDP & Regional Status)
    # "84432ENG", # Regional key figures (including GDP)
    # "71541eng", # Regional accounts (GDP per capita)
    # "82801ENG", # Regional National Accounts key figures
    # "86092NED", # Socio-economic status scores by neighborhood
    # "85900NED", # Socio-economic status (2023 index)
    
    ## Labor Market & Workforce Dynamics (SBI 2008 & Contracts)
    # "84939NED", # Employment; jobs, wages, and hours; SBI 2008; region
    # "81414NED", # Jobs of employees; gender, age, and SBI 2008
    # "82325NED", # Jobs of employees; employment type, SBI 2008
    # "82623NED", # Flexible labor; position in the labor market
    # "81408NED", # Self-employed; characteristics and income

    ## Demographics & Education (The 'Human' Factor)
    # "84671NED", # Population; gender, age, and marital status, 1 January
    # "84669NED", # Population; gender, age, and origin/background
    # "82439NED", # Educational attainment; labor market position and gender
    # "81567NED", # Level of education; population 15 to 75 years

    ## Socio-Economic Status & Regional Data
    # "83913NED", # Socio-economic category; person, gender, age, and region
    # "83648NED", # Key figures; districts and neighborhoods (Wijken en Buurten)
    # "83734NED", # Regional core figures; Netherlands (General overview)
    # "84566NED", # Income of households; characteristics of households, region
    # "83738NED", # Income of persons; key figures by region
]