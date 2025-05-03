import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = '../../Recursos/Dataset/AdultDataset/adult.csv'
#names_ = ['age','workclass','fnlwgt','education','educational-num','marital-status','occupation','relationship','race','gender','capital-gain','capital-loss','hours-per-week','native-country','income']
data = pd.read_csv(filename)

print(data)