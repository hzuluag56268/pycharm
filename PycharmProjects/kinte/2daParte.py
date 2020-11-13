from tkinter import *

from PIL import  ImageTk, Image
root = Tk()
root.title('icon exit')
root.iconbitmap('image.ico')

image1 = ImageTk.PhotoImage(Image.open('1.jpeg'))
image2 = ImageTk.PhotoImage(Image.open('2.jpeg'))
image3 = ImageTk.PhotoImage(Image.open('3.jpeg'))
list = [image1,image2,image3]

label1 = Label(image=image1)
label1.grid(row=2,column=0,columnspan=3)

statusbarl = Label(root, text='image {} out of {}'.format(1,len(list)), relief=SUNKEN, anchor=E)
statusbarl.grid(row=1,column=0, columnspan=3, sticky=W+E)

def forward_click(position):
    global label1, forwardb, backb, statusbarl
    print(position)
    label1.grid_forget()

    label1= Label(image=list[position])
    label1.grid(row=2,column=0,columnspan=3)

    forwardb = Button(root, text='next',command=lambda: forward_click(position + 1))
    backb = Button(root, text='back', command=lambda: forward_click(position-1) )

    forwardb.grid(row=0,column=2)
    backb.grid(row=0, column=1)

    if position == 2:
        forwardb = Button(root, text='next', state=DISABLED)
        forwardb.grid(row=0,column=2)

    if position == 0:
        backb = Button(root, text='back', state=DISABLED )
        backb.grid(row=0, column=1)

    statusbarl = Label(root, text='image {} out of {}'.format(position + 1, len(list)))
    statusbarl.grid(row=1, column=0, columnspan=3)
exitb = Button(root, text='salir', command=root.quit)
forwardb = Button(root, text='next',command=lambda: forward_click(1))
backb = Button(root, text='back', state=DISABLED)

exitb.grid(row=0,column=0)
forwardb.grid(row=0,column=2)
backb.grid(row=0,column=1)

root.mainloop()