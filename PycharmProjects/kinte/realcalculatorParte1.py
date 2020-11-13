from tkinter import *
import numpy as np
root = Tk()
root.title('calculator')
root.iconbitmap(r'C:\Users\Usuario\PycharmProjects\image.ico')

number = 0
e = Entry(root, width=35 , borderwidth=5)
e.grid(row=0,column=0,columnspan=3, padx=1,pady=1)

def bclick(digit):
    e.insert(END, digit)

def badd_click(operationType):
    global first
    global operation
    operation = operationType
    first = e.get()
    e.delete(0,END)

def bres_click():
    second = e.get()
    e.delete(0, END)
    global  number
    if operation == 'sum':
        res= int(first) + int(second)
    elif operation == 'multiply':
        res= int(first) * int(second)

    list = ['te ganaste {} culiadas'.format(res), 'hoy vamos por {} metidas por el culito'.format(res),
            'que tal {} veces por la garganta'.format(res), 'y que tal una contada hasta {} '.format(res) ]

    e.insert(0,list[number])
    number = np.random.randint(0, 4)

def bclear_click():
    e.delete(0,END)

b7 = Button(root, text='7',padx=30,pady=20,command= lambda : bclick(7) ).grid(row=1,column=0)
b8 = Button(root, text='8',padx=30,pady=20,command= lambda : bclick(8) ).grid(row=1,column=1)
b9 = Button(root, text='9',padx=30,pady=20,command= lambda : bclick(9) ).grid(row=1,column=2)

b4 = Button(root, text='4',padx=30,pady=20,command= lambda : bclick(4) ).grid(row=2,column=0)
b5 = Button(root, text='5',padx=30,pady=20,command= lambda : bclick(5) ).grid(row=2,column=1)
b6 = Button(root, text='6',padx=30,pady=20,command= lambda : bclick(6) ).grid(row=2,column=2)

b1 = Button(root, text='1',padx=30,pady=20,command= lambda : bclick(1) ).grid(row=3,column=0)
b2 = Button(root, text='2',padx=30,pady=20,command= lambda : bclick(2) ).grid(row=3,column=1)
b3 = Button(root, text='3',padx=30,pady=20,command= lambda : bclick(3) ).grid(row=3,column=2)
b0 = Button(root, text='0',padx=30,pady=20,command= lambda : bclick(0) ).grid(row=4,column=1)

badd = Button(root, text='suma',padx=19,pady=20,command= lambda: badd_click('sum') ).grid(row=4,column=0)

bmultiply = Button(root, text='multiplicar',padx=4,pady=20,command=  lambda: badd_click('multiply') ).grid(row=4,column=2)


bres = Button(root, text='=',padx=30,pady=20,command= bres_click ).grid(row=5,column=0)
bclear = Button(root, text='Nueva Operacion',padx=22,pady=20,command= bclear_click ).grid(row=5,column=1,columnspan=2)

root.mainloop()
print(5)


