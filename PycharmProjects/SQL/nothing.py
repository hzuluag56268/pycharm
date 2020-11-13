from tkinter import *

root = Tk()
frame = Frame(root)
frame.pack()
frame2 = Frame(root)
def hide():
    frame.pack_forget()
    bottomframe.pack_forget()
    frame2.pack()

def home():
    frame2.pack_forget()
    frame.pack()
    bottomframe.pack(side=BOTTOM)
bottomframe = Frame(root)
bottomframe.pack( side = BOTTOM )

redbutton = Button(frame, text="Red", fg="red")
redbutton.pack( side = LEFT)

greenbutton = Button(frame, text="Brown", fg="brown")
greenbutton.pack( side = LEFT )

bluebutton = Button(frame, text="Blue", fg="blue")
bluebutton.pack( side = LEFT )

blackbutton = Button(bottomframe, text="Black", fg="black",command=hide)
blackbutton.pack( side = BOTTOM)

yellowbutton = Button(frame2, text="yellow", fg="yellow",command=home)
yellowbutton.grid(row=1,column=1)


root.mainloop()