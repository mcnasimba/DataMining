import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_excel('./onlineRetailTmp.xlsx', sheet_name='OnlineRetail')
print(df[df["InvoiceNo"]==536365])

#    - Eliminar filas con valores faltantes relevantes (si es necesario)
df.dropna(subset=['InvoiceNo', 'StockCode'], inplace=True)

#    - Eliminar espacios en blanco al inicio y al final de los códigos de los productos
df['StockCode'] = df['StockCode'].str.strip()

#    - Eliminar transacciones canceladas (códigos de factura que empiezan con 'C')
df = df[~df['InvoiceNo'].astype(str).str.contains('C')]

#    - Filtrar por país si es necesario para reducir la complejidad (ejemplo: Reino Unido)
df = df[df['Country'] == 'United Kingdom']

#    - Convertir la columna Quantity a entero (puede haber valores negativos por devoluciones)
df = df[df['Quantity'] > 0]



#    - Agrupar los ítems por número de factura y crear una lista de los productos comprados en cada transacción
basket = df.groupby(['InvoiceNo'])['StockCode'].apply(list).to_list()

#    - Crear una representación booleana de las transacciones (formato requerido por mlxtend)
te = TransactionEncoder()
te_ary = te.fit(basket).transform(basket)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)



#    - Definir el soporte mínimo (ajustar según sea necesario)
support_min = 0.01
frequent_itemsets = apriori(df_encoded, min_support=support_min, use_colnames=True)


#    - Definir la métrica y el umbral mínimo (ejemplo: confianza mínima)
confidence_min = 0.1
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence_min)



print("\nConjuntos de Ítems Frecuentes:")
print(frequent_itemsets.head())

print("\nReglas de Asociación:")
print(rules.head())

#    - Explorar las reglas ordenándolas por diferentes métricas (ejemplo: lift)
rules_sorted_by_lift = rules.sort_values(by='lift', ascending=False)
print("\nReglas de Asociación Ordenadas por Lift:")
print(rules_sorted_by_lift.head(10))

#    - Filtrar reglas basadas en métricas específicas (ejemplo: lift > 3 y confianza > 0.5)
interesting_rules = rules[(rules['lift'] > 3) & (rules['confidence'] > 0.5)]
print("\nReglas de Asociación Interesantes (lift > 3 y confianza > 0.5):")
print(interesting_rules)

# 7. Conclusiones y Posibles Aplicaciones (Esto se haría en tu informe)
#    - Interpretar las reglas encontradas en el contexto del negocio.
#    - Proponer ideas sobre cómo se podrían utilizar estas reglas (recomendaciones, promociones, etc.).



