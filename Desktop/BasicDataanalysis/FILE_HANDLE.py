# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 09:35:51 2025

@author: aya mostafa
"""

import file_handler as fh

# اقرأ csv
df = fh.read_csv(r"C:\Users\aya mostafa\Desktop\hand_file.csv.txt")

# اطبع أول 5 صفوف
print(df.head())

# صدّر لملف Excel
fh.save_excel(df, r"C:\Users\aya mostafa\Desktop\hand_file.xlsx")

print("Done")