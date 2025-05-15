import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

filename = '../../Recursos/Dataset/AdultDataset/adult.csv'
#names_ = ['age','workclass','fnlwgt','education','educational-num','marital-status','occupation','relationship','race','gender','capital-gain','capital-loss','hours-per-week','native-country','income']
data = pd.read_csv(filename)

variable_classification = {
    'age': 'Continuous',
    'workclass': 'Discrete',
    'fnlwgt': 'Continuous',
    'education': 'Discrete',
    'educational-num': 'Continuous',
    'marital-status': 'Discrete',
    'occupation': 'Discrete',
    'relationship': 'Discrete',
    'race': 'Discrete',
    'sex': 'Discrete',
    'capital-gain': 'Continuous',
    'capital-loss': 'Continuous',
    'hours-per-week': 'Continuous',
    'native-country': 'Discrete',
    'income': 'Discrete'
}

print("Variables classification")
for var, desc in variable_classification.items():
    print(f"{var}: {desc}")

# 2. Selección de variables continuas
continuous_vars = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
continuous_data = data[continuous_vars]

print(continuous_data)

descriptive_stats = continuous_data.describe()
descriptive_stats.loc['skewness'] = continuous_data.skew()
descriptive_stats.loc['kurtosis'] = continuous_data.kurtosis()
print("\nEstadísticas descriptivas:\n", descriptive_stats)

# 4. Visualización: histogramas y gráficos de densidad
sns.set(style="whitegrid")
for var in continuous_vars:
    plt.figure(figsize=(10, 5))

    # Histograma
    plt.subplot(1, 2, 1)
    sns.histplot(data[var], kde=False, bins=30)
    plt.title(f'Histograma de {var}')
    plt.xlabel(var)
    plt.ylabel('Frecuencia')

    # Gráfico de densidad
    plt.subplot(1, 2, 2)
    sns.kdeplot(data[var], color='blue')
    plt.title(f'Densidad de {var}')
    plt.xlabel(var)
    plt.ylabel('Densidad')

    plt.tight_layout()
    plt.savefig(f'{var}_visualization.png')  # Guardar para informe
    plt.show()

#print(data)