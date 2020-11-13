import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import re
'''6. Write a Python program to get a list, sorted in increasing order
 by the last element in each tuple from a given list of non-empty tuples.

Sample List : [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]

Expected Result : [(2, 1), (1, 2), (2, 3), (4, 4), (2, 5)]

#lis = [(1, 9), (6, 2), (3, 0)]
lis = [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]


#solution

def sort_list(lis):
    return sorted(lis, key= lambda x : x[1], reverse=  True)
print(sort_list(lis))
'''
"""7. Write a Python program to remove duplicates from a list. 


def remove(lis):
    return   [*set(lis)]

print(remove([1,2,1]))

"""
'''8. Write a Python program to check a list is empty or not.

def is_empty(lis):
    if  lis :
        return ' not empty'

    return ' empty'
print(is_empty([]))
'''
'''9 Write a Python program to clone or copy a list


l = [1, 4]
def copy(l):
    return l.copy()
print([x for x in l])
print(copy(l))

'''
'''11. Write a Python function that takes two lists and returns True
 if they have at least one common member.

l1 = [1,2,4]
l2 = [3,1,5]

def common(l1, l2):
    l12 = l1 + l2
    if len(l12) != len(set(l12)):
        return True
    return False
print(common(l1, l2))

def common2(l1, l2):
    for x in l1:
        for y in l2:
            if x == y:

                return True
                break
    return False
print(common2(l1, l2))'''
'''
#13. Write a Python program to generate a 3*4*6 3D array whose each element is *.

def array(l,t,y):
    array = [[['*' for e in range(y)] for i in range(t)]for u in range(l)]
    print(np.shape(array))
array(3,4,6)'''
'''#14. Write a Python program to print the numbers of
#a specified list after removing even numbers from it.

def odd(l):
    print([x for x in l if x%2 != 0])
odd([x for x in range(10)])'''
'''#16. Write a Python program to generate and print a list of first and last 5 elements
# where the values are square of numbers between 1 and 30 (both included).

def square(elements, length):
    l = [x**2 for x in range(1, length+1)]
    print(l[:elements] + [round(x, 1) for x in l[-elements:]])
square(5, 30)'''
'''#18. Write a Python program to generate all permutations of a list in Python.
import  itertools 
l = ['1', '2', '3', '4']

x =list(itertools.permutations(l))
print(x)'''
'''19. Write a Python program to get the difference between the two lists.

l1 = list(range(10))
l2 = list(range(5, 100, 2))

def diff(l1, l2):
    res = []
    for e in l1:
        if l1[e] not in l2:
            res.append(l1[e])
    for x in l2:
        if x not in l1 and x not in res:
            res.append(x)

    return res
print(diff(l1, l2))
# other way
re = set(l1).symmetric_difference(set(l2))
print(re)
'''



file = r'C:\Users\Usuario\Desktop\EDT.xlsx'
xls = pd.ExcelFile(file)
d_f = xls.parse(0)
df1 = d_f[['Nombre de tarea']]

#filter = df1['Nombre de tarea'].str.match(r'\[A-Za-z]+_')
list1 = list(df1['Nombre de tarea'])


list2 = [re.findall(r'.*\w+_', i) for i in list1]
list3 = [','.join(x) for x in list2 if x]
column_activity = []
for i in list3:
    if re.search(r'Aprobación desarrollos', i) is None:

        column_activity.append(i.strip(r'[ _]'))

index1 = range(65, 65 + len(column_activity))
id = [chr(l) for l in index1]


np.random.seed(20)
mas_probable = np.array([9, 11, 10, 23, 20, 17, 28, 22, 35, 42, 35, 48, 20, 10, 30])
optimista = mas_probable - np.random.randint(3, 5, 1)
pesimista = mas_probable + np.random.randint(3, 7, 1)
tiempo_esperado = np.round((optimista + 4 * mas_probable + pesimista) / 6)

#print(optimista, '\n', pesimista, '\n', tiempo_esperado)

precedencia = ['-', 'A', 'B', 'B', 'D, C', 'D', 'D', 'G', 'H',
               'H', 'H', 'H', 'I,J,K,L', 'M,E,F', 'N']
#df['Precedencia'] = precedencia
#print(df)

tabla_tiempos = pd.DataFrame({'ID' : id, 'Actividad' : column_activity,
                   'Optimista' : optimista, 'Mas probable' : mas_probable,
                    'Pesimista' : pesimista, 'Tiempo Esperado_dias' : tiempo_esperado})
print(tabla_tiempos)
tabla_tiempos.to_excel(r'C:\Users\Usuario\Desktop\tabla_tiempos.xlsx')

tabla_precedencia = pd.DataFrame({'ID' : id, 'Actividad' : column_activity,
                                  'Tiempo Esperado_dias': tiempo_esperado,
                                  'Precedencia' : precedencia})
print(tabla_precedencia)
#tabla_precedencia.to_excel(r'C:\Users\Usuario\Desktop\tabla_precedencia.xlsx')


