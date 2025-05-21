import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_excel("sample_data/OnlineRetail.xlsx", sheet_name='OnlineRetail')
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df = df[df['Country'] == 'United Kingdom']  # Puedes cambiar esto si quieres analizar otros países

print (df)

# 🧺 Crear matriz de transacciones tipo cesta
basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
basket = basket.map(lambda x: 1 if x > 0 else 0)

# Filtrar productos que aparecen en al menos 50 transacciones
basket_reduced = basket.loc[:, (basket.sum(axis=0) > 50)]

# ✅ Aplicar Apriori
frequent_items = apriori(basket, min_support=0.05, use_colnames=True)

#rules = association_rules(frequent_items, metric="lift", min_threshold=1)
#rules = rules[(rules['lift'] > 1.5) & (rules['confidence'] > 0.4)]

rules = association_rules(frequent_items, metric='lift', min_threshold=1.5)
rules = rules[(rules['confidence'] >= 0.5)]
rules = rules.sort_values(by='lift', ascending=False)

# 👁️ Ver las 10 reglas principales
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)

from matplotlib import pyplot as plt
import seaborn as sns
_df_2.groupby('antecedents').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[['top', 'right',]].set_visible(False)

import matplotlib.pyplot as plt

# 🔟 Top 10 reglas por lift
top10 = rules.nlargest(10, 'lift')

# 📉 Gráfico de barras
plt.figure(figsize=(10,6))
plt.barh(range(len(top10)), top10['lift'], color='skyblue')
plt.yticks(range(len(top10)), [f"{list(a)} → {list(c)}" for a, c in zip(top10['antecedents'], top10['consequents'])])
plt.xlabel("Lift")
plt.title("Top 10 Reglas de Asociación por Lift")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()