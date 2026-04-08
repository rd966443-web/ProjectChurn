import sqlite3

#connect database
def get_connection():
    con=sqlite3.connect("dbchurn.db")
    return con

#create table
def create_table():
    con=get_connection()
    cur=con.cursor()
    cur.execute("""
    create table if not exists data_overview(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Gender TEXT,
        Age INTEGER,
        SeniorCitizen INTEGER,
        Partner TEXT,
        Dependents TEXT,
        Tenure INTEGER,
        PhoneService TEXT,
        MultipleLines TEXT,
        InternetService TEXT,
        OnlineSecurity TEXT,
        OnlineBackup TEXT,
        DeviceProtection TEXT,
        TechSupport TEXT,
        StreamingTV TEXT,
        StreamingMovies TEXT,            
        Contract TEXT,            
        PaperlessBilling TEXT,
        PaymentMethod TEXT,
        MonthlyCharges REAL,
        TotalCharges REAL,
        Prediction INTEGER,
        Probability REAL
    )
    """)
    con.commit()
    con.close()
    return "Table created successfully"

def insert_data(data,prediction,probability):
    try:
        con=get_connection()
        cur=con.cursor()
        cur.execute("""
        INSERT INTO data_overview (
            Gender,Age,SeniorCitizen,Partner,Dependents,Tenure,
            PhoneService,MultipleLines,InternetService,
            OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,
            StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,
            MonthlyCharges,TotalCharges,Prediction,Probability)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,(
            data["Gender"], 
            data["Age"], 
            data["SeniorCitizen"], 
            data["Partner"], 
            data["Dependents"], 
            data["Tenure"],
            data["PhoneService"], 
            data["MultipleLines"], 
            data["InternetService"],
            data["OnlineSecurity"], 
            data["OnlineBackup"], 
            data["DeviceProtection"], 
            data["TechSupport"],
            data["StreamingTV"], 
            data["StreamingMovies"], 
            data["Contract"], 
            data["PaperlessBilling"], 
            data["PaymentMethod"],
            data["MonthlyCharges"], 
            data["TotalCharges"], 
            prediction, 
            probability
        ))
        con.commit()
    except Exception as e:
        print("DB Error:", e)
    finally:
        con.close()

                
    
    