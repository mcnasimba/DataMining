import pandas as pd

datafile = pd.read_excel('./onlineRetailTmp.xlsx', sheet_name='OnlineRetail')
print(datafile[datafile["InvoiceNo"]==536365])
