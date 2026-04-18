import sqlite3
import pandas as pd
import os

import hashlib
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
# SHA-256

#connect database
def get_connection():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "dbchurn.db")
    con=sqlite3.connect(db_path, check_same_thread=False, timeout=10)
    return con

#create data table
def create_table():
    con=get_connection()
    cur=con.cursor()
    cur.execute("""
    create table if not exists data_overview(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
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
    # Index for faster queries
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_username 
    ON data_overview(username)
    """)
    con.commit()
    con.close()
    return "Table created successfully"

#create user table
def create_user_table():
    con=get_connection()
    cur=con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)
    con.commit()
    con.close()

#insert data
def insert_data(username,data,prediction,probability):
    con=None
    try:
        con=get_connection()
        cur=con.cursor()
        cur.execute("""
        INSERT INTO data_overview (
            username,Gender,Age,SeniorCitizen,Partner,Dependents,Tenure,
            PhoneService,MultipleLines,InternetService,
            OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,
            StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,
            MonthlyCharges,TotalCharges,Prediction,Probability)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,(
            username,
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
        raise Exception(e)
    finally:
        if con:
            con.close()

#fetch all data
def fetch_data(username):
    con = get_connection()
    try:
        query = "SELECT * FROM data_overview WHERE username = ? ORDER BY id DESC"
        df = pd.read_sql(query, con, params=(username,))
        return df
    finally:
        con.close()

#delete data
def delete_data(username, customer_id):
    con = get_connection()
    try:
        cur = con.cursor()
        cur.execute(
            "DELETE FROM data_overview WHERE id = ? AND username = ?",
            (customer_id, username)
        )
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()

#search data
def search_data(username, keyword):
    con = get_connection()
    try:
        query = """
        SELECT * FROM data_overview
        WHERE username = ?
        AND (
            LOWER(Gender) LIKE LOWER(?) OR 
            LOWER(PaymentMethod) LIKE LOWER(?) OR 
            LOWER(Contract) LIKE LOWER(?) OR 
            LOWER(InternetService) LIKE LOWER(?)
        )
        ORDER BY id DESC
            """
        df = pd.read_sql(
            query,
            con,
            params=(username, f"%{keyword}%", f"%{keyword}%", f"%{keyword}%", f"%{keyword}%")
        )
        return df
    finally:
        con.close()

#stats data->gives summary
def get_stats(username):
    con = get_connection()
    try:
        cur = con.cursor()
      
        # total records
        cur.execute("SELECT COUNT(*) FROM data_overview WHERE username=?", (username,))
        total = cur.fetchone()[0]

        # average probability
        cur.execute("SELECT AVG(Probability) FROM data_overview WHERE username=?", (username,))
        avg_prob = cur.fetchone()[0]
        if avg_prob is None:
            avg_prob = 0

        # high risk count
        cur.execute("""
        SELECT COUNT(*) FROM data_overview 
        WHERE Prediction = 1 AND username=?
        """, (username,))
        high_risk = cur.fetchone()[0]
        return total, avg_prob, high_risk
    finally:
        con.close()

#update data
def update_prediction(id, username, prediction, probability):
    con = get_connection()
    try:
        cur = con.cursor()
        cur.execute("""
            UPDATE data_overview
            SET Prediction = ?, Probability = ?
            WHERE id = ? AND username = ?
        """, (prediction, probability, id, username))
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()

#signup function
def signup_user(username, password):
    con = get_connection()
    try:
        cur = con.cursor()
        hashed = hash_password(password)
        cur.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed)
        )
        con.commit()
        return True
    except:
        return False
    finally:
        con.close()

#login function
def login_user(username, password):
    con = get_connection()
    try:
        cur = con.cursor()
        hashed = hash_password(password)
        cur.execute(
            "SELECT * FROM users WHERE username = ? AND password = ?",
            (username, hashed)
        )
        user = cur.fetchone()
        return user is not None
    finally:
        con.close() 








