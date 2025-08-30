# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 09:53:32 2025

@author: aya mostafa
"""

import preprocessing as pp

# حمل البيانات
df = pp.load_data("C:/Users/aya mostafa/Downloads/Data_preprossising.csv")

# لو الملف اتحمل صح
if df is not None:
    pp.check_dtypes(df)
    df = pp.convert_dtypes(df)  # يحول الداتا تايبس
    pp.check_missing_values(df)  # يطبع القيم الناقصة
    df = pp.handle_missing_values(df, method="mean")  # يعالج القيم الناقصة