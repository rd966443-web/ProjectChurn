import sqlite3
import pandas as pd

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

#insert data
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
        if con:
            con.close()

#fetch all data
def fetch_data():
    con = get_connection()
    df = pd.read_sql("SELECT * FROM data_overview", con)
    con.close()
    return df

#delete data
def delete_data(customer_id):
    con = get_connection()
    cur = con.cursor()
    cur.execute("DELETE FROM data_overview WHERE id = ?", (customer_id,))
    con.commit()
    con.close()

#serch data
def search_data(keyword):
    con = get_connection()
    cur = con.cursor()

    query = """
    SELECT * FROM data_overview
    WHERE Gender LIKE ? OR PaymentMethod LIKE ?
    """

    cur.execute(query, (f"%{keyword}%", f"%{keyword}%"))
    rows = cur.fetchall()

    con.close()
    return rows

#stats data->gives summary
def get_stats():
    con = get_connection()
    cur = con.cursor()
    # Total records
    cur.execute("SELECT COUNT(*) FROM data_overview")
    total = cur.fetchone()[0]
    # Average probability
    cur.execute("SELECT AVG(Probability) FROM data_overview")
    avg_prob = cur.fetchone()[0]
    # High risk customers
    cur.execute("SELECT COUNT(*) FROM data_overview WHERE Prediction = 1")
    high_risk = cur.fetchone()[0]
    con.close()
    return total, avg_prob, high_risk

#update data
def update_prediction(customer_id, prediction, probability):
    con = get_connection()
    cur = con.cursor()

    cur.execute("""
    UPDATE data_overview
    SET Prediction = ?, Probability = ?
    WHERE id = ?
    """, (prediction, probability, customer_id))

    con.commit()
    con.close()



                
    
    