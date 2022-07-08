import csv
import pandas as pd
from pathlib import Path

# data to write
data = [['Paris', 150], ['London', 200]]
header_row = ['office_name', 'num_employees']

df = pd.read_csv("Data1d.csv", delimiter='\t')
print(df.info())
# print(df["office_name"])
# print(df["num_employees"])

# write into the csv file
with open("Data1d.csv", 'a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter='\t')
    # csv_writer.writerow(header_row)
    for row in data:
        csv_writer.writerow(row)

df = pd.read_csv("Data1d.csv", delimiter='\t')
print(df.info())
