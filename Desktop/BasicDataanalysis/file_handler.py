# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 09:39:01 2025

@author: aya mostafa
"""
import pandas as pd
import sqlite3

# ---------- قراءة الملفات ----------
def read_csv(path):
    return pd.read_csv(path)

def read_excel(path):
    return pd.read_excel(path)

def read_json(path):
    return pd.read_json(path)

def read_db(path, table_name):
    conn = sqlite3.connect(path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

# ---------- حفظ الملفات ----------
def save_csv(df, path):
    df.to_csv(path, index=False)

def save_excel(df, path):
    df.to_excel(path, index=False)

def save_json(df, path):
    df.to_json(path, orient="records", indent=4)

def save_db(df, path, table_name):
    conn = sqlite3.connect(path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
