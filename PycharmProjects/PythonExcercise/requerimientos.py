'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import re

x = pd.read_csv('datasets/logins.csv')
print(x.head())'''
'''attributes = ['Primer Nombre', 'Primer Apellido', 'Cedula', 'telefono', 'Direccion', 'Apto', 'Conjunto', 'Placa', 'Marca', 'Modelo']
attribute_list = (' ').join([e + ': \n' for e in attributes])
x = (',').strip(attribute_list)
print(type(attribute_list))
print(attribute_list)
print(x)'''

x= ['one', 'two']
y= [68, 69]

z = {i:j for i, j in zip(x,y)}
print(type(z))