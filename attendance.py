from flask import Flask, render_template, request, redirect, url_for, flash
import psycopg2 #pip install psycopg2 
import psycopg2.extras
from datetime import datetime
DB_HOST = "localhost"
DB_NAME = "SampleDB"
DB_USER = "postgres"
DB_PASS = "ROOT"
 
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)

time_now = datetime.now()
tStr = time_now.strftime('%H:%M:%S')
dStr = time_now.strftime('%d_%m_%Y') 
attendance_sheet = "Attendance_" + dStr 

cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
def making_table():
    cur.execute(f"CREATE TABLE IF NOT EXISTS {attendance_sheet} (name VARCHAR ( 40 ) PRIMARY KEY,time VARCHAR ( 40 ) ,date VARCHAR ( 40 ));")
    conn.commit()
    cur.execute("ROLLBACK")
    conn.commit()
 
def add_attendance(name , tStr = None , dStr = None):
    making_table()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    name = str(name)
    name = name.capitalize()
    time_now = datetime.now()
    if tStr == '':
        tStr = time_now.strftime('%H:%M:%S')
    if dStr == '':
        dStr = time_now.strftime('%d_%m_%Y') 
    try:
        cur.execute(f"INSERT INTO {attendance_sheet} (name, time, date) VALUES (%s,%s,%s)", (name, tStr, dStr))
        conn.commit()
    except:
        cur.execute("ROLLBACK")
        conn.commit()    
        
