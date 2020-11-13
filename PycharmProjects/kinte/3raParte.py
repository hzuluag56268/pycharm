'''
radioButton,   LabelFrame
'''


from tkinter import *
from PIL import  ImageTk, Image
root = Tk()
root.title('icon exit')
root.iconbitmap('image.ico')

frame = LabelFrame(root,text='this is a frame',padx=100,pady=100)
frame.grid(padx=20,pady=20)

la= Label(frame,text='selecciona').pack()


def radio_click(message, messagefromButton=''):
    showOptionL = Label(frame, text=message + messagefromButton).pack()

var = StringVar()
var.set('u')

xy = [( 'numero 1','ill fuck'),   ( 'numero 2','u')]
for text, value in xy:
    radio = Radiobutton(frame,text=text, value=value, variable=var,command=lambda: radio_click(var.get())).pack()

selectRadio = Button(frame,text='select',command=lambda :radio_click(var.get(), 'selected by button')).pack()

root.mainloop()