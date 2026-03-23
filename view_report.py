import pandas as pd

df = pd.read_csv(r"C:\Users\devqu\Downloads\parkinson_project\data\dataset_final\dataset_report.csv")

df.to_excel("dataset_report.xlsx", index=False)

print("Saved dataset_report.xlsx")