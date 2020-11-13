'''

'''



from tkinter import *
from tkinter import messagebox
root = Tk()
root.title('icon exit')
root.iconbitmap('image.ico')


def create():

    answer = messagebox.askyesno('create','wanna create')
    if answer:
        window = Toplevel()
        root.title('window')
        l = Label(window,text='heres the window').pack()
        closeB = Button(window,text='close',command=window.destroy).pack()

Button(root,text='create window',command=create).pack()



root.mainloop()