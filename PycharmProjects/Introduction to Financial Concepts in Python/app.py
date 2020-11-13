import numpy as np
import pandas as pd


class App:
    def __init__(self,home_value=200,annual_int_rate=11,duration_years=15,
                         inicial_pay_percent=20, annual_inflation_rate=3):
        self.home_value = home_value
        self.inicial_pay_percent = inicial_pay_percent / 100
        self.annual_int_rate = annual_int_rate / 100
        self.duration_years = duration_years
        self.annual_inflation_rate = annual_inflation_rate/100


    def monthly_payments(self):


        self.loan = self.home_value * (1-self.inicial_pay_percent)
        self.inicial_payment = self.home_value * self.inicial_pay_percent
        self.monthly_int_rate = (1 + self.annual_int_rate)**(1/12)-1
        self.duration_months = 12 * self.duration_years
        self.monthly_payment = -np.pmt(self.monthly_int_rate,self.duration_months,pv=self.loan)
        answer = f'==================================================================== \n' \
                 f' House value: {self.home_value} millones.' \
                 f'   {self.inicial_pay_percent*100}% for the initial_payment:   {self.inicial_payment} millones\n' \
                 f'loan: {self.loan}    monthly payment: ||||  {round(self.monthly_payment,3)} ||||  millions\n' \
                 f'==================================================================== \n'
        return answer


    def total_paid(self):

        self.monthly_inflation_rate = (1 + self.annual_inflation_rate)**(1/12)-1
        self.paid_total = self.monthly_payment * self.duration_months
        self.discounted_value = np.npv(self.monthly_inflation_rate,
                                       np.repeat(self.monthly_payment,self.duration_months)
                                       )
        self.times_paid = self.paid_total / self.loan
        self.times_paid_discounted = round(self.discounted_value / self.loan,2)
        answer = f'==================================================================== \n' \
                 f'house value: {self.home_value} millones, ' \
                 f' initial payment: {self.inicial_pay_percent*100}%     {self.inicial_payment} millones  \n' \
                 f'  loan of {self.loan} millones' \
                 f'. after {self.duration_years} years total paid: {round(self.paid_total,3)} millones,  {round(self.times_paid, 2)} times more \n' \
                 f'Preset value: {round(self.discounted_value,3)},  {self.times_paid_discounted} times more \n' \
                 f'===================================================================='

        return answer

    def interest_paid(self,a_rent_growth, rent=0):
        if rent == 0:
            self.rent = round(self.monthly_payment,3)
        else:
            self.rent = rent

        self.a_rent_growth = a_rent_growth/100
        self.m_rent_growth = (1 + self.a_rent_growth)**(1/12)-1

        self.projected_rent = self.rent * np.cumprod(1+np.repeat(self.m_rent_growth,self.duration_months))
        self.cumulative_rent = np.cumsum(self.projected_rent)
        self.interests_paid = self.cumulative_rent > (self.paid_total - self.loan)
        self.interests_paid_in_years = round((pd.value_counts(self.interests_paid)[0] / 12),1)
        answer = f'De los {self.duration_years} years, a los ' \
                 f'{self.interests_paid_in_years} years the interest is paid with {self.a_rent_growth *100}% anual rent growth\n' \
                 f'with rent of {self.rent} millions'
        return  answer

    def create_df(self):

        def monthly_payments(self):
            self.loan = self.home_value * (1 - self.inicial_pay_percent)
            self.inicial_payment = self.home_value * self.inicial_pay_percent
            self.monthly_int_rate = (1 + self.annual_int_rate) ** (1 / 12) - 1
            self.duration_months = 12 * self.duration_years
            self.monthly_payment = -np.pmt(self.monthly_int_rate, self.duration_months, pv=self.loan)
            self.answer = f'==================================================================== \n' \
                     f' House value: {self.home_value} millones.' \
                     f'   initial_payment: {self.inicial_pay_percent * 100}% ,  {self.inicial_payment} millones\n' \
                     f'loan: {self.loan}    monthly payment: ||||  {round(self.monthly_payment, 3)} ||||  millions\n' \
                     f'==================================================================== \n'

            def total_paid(self):
                self.monthly_inflation_rate = (1 + self.annual_inflation_rate) ** (1 / 12) - 1
                self.paid_total = self.monthly_payment * self.duration_months
                self.discounted_value = np.npv(self.monthly_inflation_rate,
                                               np.repeat(self.monthly_payment, self.duration_months)
                                               )
                self.times_paid = self.paid_total / self.loan
                self.times_paid_discounted = round(self.discounted_value / self.loan, 2)
                answer = f'==================================================================== \n' \
                         f'house value: {self.home_value} millones, ' \
                         f' initial payment: {self.inicial_pay_percent * 100}%     {self.inicial_payment} millones  \n' \
                         f'  loan of {self.loan} millones' \
                         f'. after {self.duration_years} years total paid: {round(self.paid_total, 3)} millones,  {round(self.times_paid, 2)} times more \n' \
                         f'Preset value: {round(self.discounted_value, 3)},  {self.times_paid_discounted} times more \n' \
                         f'===================================================================='

                return pd.DataFrame([self.home_value,self.inicial_pay_percent*100,self.inicial_payment,
                                self.loan, self.annual_int_rate*100,
                                self.monthly_payment, self.paid_total,self.times_paid,
                                self.annual_inflation_rate*100,self.discounted_value,self.times_paid_discounted
                                ],
                               columns=[str(self.duration_years) + ' years'],
                               index=['Home_Value', 'Initial%',' Initial_pmt','Loan','TEA',
                                      'Monthly_pmt','Total_paid','times_paid','Ave_inflation',
                                      'Today_value','today_times'
                                      ]
                               )


            return total_paid(self)



        return monthly_payments(self)



class Concatenate_df:
    def __init__(self,*dfs):
        self.dfs = dfs


    def concatenate(self):
        self.new_df =pd.concat(self.dfs,axis=1)


        return self.new_df





'''month_payment = App(159.375,11.5,20,20,4)
payment = month_payment.monthly_payments()
test = month_payment.total_paid()
interest_paid = month_payment.interest_paid(1)
df = month_payment.create_df()
print(payment)
print(test)
print(interest_paid)
print(df) '''         #tester with reliable values


'''calculation1 = App(159.375,11.5,20,20,4)
print(calculation1.monthly_payments())
df1 = calculation1.create_df()
calculation2 = App(159.375,11.5,30,20,4)
calculation3 = App(159.375,11.5,15,20,4)
df2 = calculation2.create_df()
df3 = calculation3.create_df()


together = Concatenate_df(*[df3,df1,df2])
dfall = together.concatenate()
print(dfall)''' #tester 2











from tkinter import *
from tkinter import messagebox

root = Tk()
root.title('Quiubo Gono...')
root.iconbitmap(r'C:\Users\Usuario\PycharmProjects\image.ico')

number = 1
apts = {}    #objects created
menu_values  = []



frame1 = Frame(root,bd=10,relief=RIDGE,bg='dark slate gray')
frame1.pack()
frame2 = Frame(root,bd=10,relief=RIDGE,bg='gold3')
frame3 = Frame(root,bd=10,relief=RIDGE,bg='gold3')
frame4 = Frame(root,bd=10,relief=RIDGE,bg='gold3')
frame5 = Frame(root,bd=10,relief=RIDGE,bg='gold3')
#functions

def click_save():
    #global initial , inflation, tea, duration, value
    if e_homevalue.get() == '':
        pass

    global number, apts, menu_values, apartamento_created, rent

    #e_homevalue.delete(0,END)
    #e_plazo.delete(0,END)
    #e_TEA.delete(0,END)
    #e_inicial.delete(0,END)
    #e_inflation.delete(0,END)
    
    
    #value = e_homevalue.insert(END,159.375)
    #duration = e_plazo.insert(END,20)
    #tea = e_TEA.insert(END,11.5)
    #initial = e_inicial.insert(END,20)
    #inflation = e_inflation.insert(END,3.5)
    #nombre_proyecto = e_nombre_proyecto.insert(END,'ventus703')

    value = e_homevalue.get()
    duration = e_plazo.get()
    tea = e_TEA.get()
    initial = e_inicial.get()
    inflation = e_inflation.get()
    nombre_proyecto = e_nombre_proyecto.get()
    rent = e_rent.get()

    b_calculos.grid(row=7, column=2, padx=20, pady=20)
    b_save['text'] = 'save proyecto'+ str(number)
    apartamento_created = App(float(value),float(tea),float(duration),float(initial),float(inflation))
    apts[nombre_proyecto] = apartamento_created
    menu_values.append(nombre_proyecto)
    number += 1

 #aptsss = { var.get: {nombre_del_metodo: }}
    e_homevalue.delete(0,END)
    e_plazo.delete(0,END)
    e_TEA.delete(0,END)
    e_inicial.delete(0,END)
    e_inflation.delete(0,END)
    e_rent.delete(0,END)
    e_nombre_proyecto.delete(0,END)
    # duration = e_plazo.insert(END,20)
    # tea = e_TEA.insert(END,11.5)
    # initial = e_inicial.insert(END,20)
    # inflation = e_inflation.insert(END,3.5)
def click_calculos():
    global length_apts, menu_values, var, o_houses
    frame1.pack_forget()
    frame2.pack()

    var = StringVar()
    var.set('selecciona el proyecto')
    o_houses = OptionMenu(frame2, var, *menu_values )
    o_houses.config(bg='red',fg='white',activebackground='chocolate3')
    o_houses.grid(row=0,column=0,padx=5,pady=10)



l_homevalue = Label(frame1,text='Home_value :',font=30)
e_homevalue = Entry(frame1,font=30,relief=SUNKEN)
l_homevalue.grid(row=0,column=0,padx=20,pady=10)
e_homevalue.grid(row=0,column=1,columnspan=3,padx=20,pady=10)
l_homevalue2 = Label(frame1,text='Millones',font=30,bg='dark slate gray')
l_homevalue2.grid(row=0,column=4,padx=5,pady=10)

l_TEA = Label(frame1,text='TEA :',font=30)
e_TEA = Entry(frame1,font=30,relief=SUNKEN)
l_TEA.grid(row=1,column=0,padx=20,pady=10)
e_TEA.grid(row=1,column=1,columnspan=3,padx=20,pady=10)
l_TEA1= Label(frame1,text='% anual',font=30,bg='dark slate gray')
l_TEA1.grid(row=1,column=4,padx=5,pady=10)

l_plazo = Label(frame1,text='Plazo :',font=30)
e_plazo = Entry(frame1,font=30,relief=SUNKEN)
e_plazo.insert(END,20)
l_plazo.grid(row=2,column=0,padx=20,pady=10)
e_plazo.grid(row=2,column=1,columnspan=3,padx=20,pady=10)
l_plazo1= Label(frame1,text='Years',font=30,bg='dark slate gray')
l_plazo1.grid(row=2,column=4,padx=5,pady=10)

l_inicial = Label(frame1,text='% de la inicial :',font=30)
e_inicial = Entry(frame1,font=30,relief=SUNKEN)
e_inicial.insert(END,20)
l_inicial.grid(row=3,column=0,padx=20,pady=10)
e_inicial.grid(row=3,column=1,columnspan=3,padx=20,pady=10)
l_inicial1= Label(frame1,text='%',font=30,bg='dark slate gray')
l_inicial1.grid(row=3,column=4,padx=5,pady=10)

l_inflation = Label(frame1,text='inflacion :',font=30)
e_inflation = Entry(frame1,font=30,relief=SUNKEN)
e_inflation.insert(END,3.5)
l_inflation.grid(row=4,column=0,padx=20,pady=10)
e_inflation.grid(row=4,column=1,columnspan=3,padx=20,pady=10)
l_inflation1= Label(frame1,text='% anual',font=30,bg='dark slate gray')
l_inflation1.grid(row=4,column=4,padx=5,pady=10)

l_rent = Label(frame1,text='incremento renta :',font=30)
e_rent = Entry(frame1,font=30,relief=SUNKEN)
e_rent.insert(END,1)
l_rent.grid(row=5,column=0,padx=20,pady=10)
e_rent.grid(row=5,column=1,columnspan=3,padx=20,pady=10)
l_rent1= Label(frame1,text='% anual',font=30,bg='dark slate gray')
l_rent1.grid(row=5,column=4,padx=5,pady=10)

l_nombre_proyecto = Label(frame1,text='nombre proyecto :',font=30)
e_nombre_proyecto = Entry(frame1,font=30,relief=SUNKEN)
l_nombre_proyecto.grid(row=6,column=0,padx=20,pady=10)
e_nombre_proyecto.grid(row=6,column=1,columnspan=3,padx=20,pady=10)



b_save = Button(frame1,text='save proyecto1',font=40,borderwidth=10,
                activebackground='chocolate3',command=click_save)
b_save.grid(row=7,column=0,padx=20,pady=20)


b_calculos = Button(frame1,text='Hacer calculos',font=40,borderwidth=10,
                activebackground='chocolate3',command=click_calculos)

#calculos page frame2

#functions



def monthly_pmt():
    b_total_paid['state'] = ACTIVE
    b_interest_paid['state'] = ACTIVE
    o_houses.config(bg='black', fg='white', activebackground='chocolate3')
    textwidget.delete(1.0, END)
    mp = apts[var.get()].monthly_payments()
    textwidget.insert(END, mp)
def click_total_paid():
    textwidget.delete(1.0, END)
    tpp = apts[var.get()].total_paid()
    textwidget.insert(END, tpp)

def click_interest_paid():
    textwidget.delete(1.0, END)
    tpp = apts[var.get()].interest_paid(int(rent))
    textwidget.insert(END, tpp)


def click_display_df():
    global tpdf
    if var.get() == 'selecciona el proyecto':
        messagebox.showinfo('dsf','select project')
    else:
        frame2.pack_forget()
        frame3.pack()
        textwidget.delete(1.0, END)
        textwidget2.delete(1.0, END)

        tpdf = apts[var.get()].create_df()
        textwidget2.insert(END, tpdf)

def click_clear():
    textwidget.delete(1.0,END)
def click_add():
    frame2.pack_forget()
    frame1.pack()
def click_comparison():
    global name, Rb_select, p
    frame2.pack_forget()
    frame4.pack()
    p = StringVar()
    for row, apt in enumerate(menu_values):
        name = 'rb_select'+ str(row)
        name = Radiobutton(frame4, text=apt, variable=p, value=apt)
        name.grid(row=row+1, column=1, padx=20, pady=2,sticky=W)

b_add = Button(frame2,text='ADD more',font=10,borderwidth=5,
                activebackground='chocolate3',command=click_add,bg='gold2')
b_add.grid(row=0,column=2,padx=20,pady=5)

b_monthly_pmt = Button(frame2,text='cuota',font=40,borderwidth=10,
                activebackground='chocolate3',command=monthly_pmt,bg='dark slate gray')
b_monthly_pmt.grid(row=2,column=0,padx=20,pady=20)

b_total_paid = Button(frame2,text='Total paid',font=40,borderwidth=10, state=DISABLED,
                activebackground='dark slate gray',command=click_total_paid,bg='dark slate gray')
b_total_paid.grid(row=2,column=1,padx=20,pady=20)


b_interest_paid= Button(frame2,text='interests paid',font=40,borderwidth=10, state=DISABLED,
                activebackground='dark slate gray',command=click_interest_paid,bg='dark slate gray')
b_interest_paid.grid(row=3,column=0,padx=20,pady=20)

b_display_df= Button(frame2,text='display table',font=40,borderwidth=10,
                activebackground='chocolate3',command=click_display_df,bg='dark slate gray')
b_display_df.grid(row=3,column=1,padx=20,pady=20)

b_comparison= Button(frame2,text='compare Project',font=40,borderwidth=10,
                activebackground='chocolate3',command=click_comparison,bg='dark slate gray')
b_comparison.grid(row=3,column=2,padx=20,pady=20)

b_clear= Button(frame2,text='clear',font=15,borderwidth=2,
                activebackground='chocolate3',command=click_clear,bg='dark slate gray')
b_clear.grid(row=3,column=3,padx=20,pady=20)




textwidget = Text(frame2,bg='gold3',font=("Comic Sans MS", 12), padx=0, pady=0,width=60, height=6)
textwidget.grid(row=5, column=0,columnspan=4)



#frame3   display 1 df

def click_back():
    frame3.pack_forget()
    frame2.pack()

def click_export():


    tpdf.to_excel(r'C:\Users\Usuario\Desktop\comparasiones.xlsx')
    messagebox.showinfo('dd','done')



textwidget2 = Text(frame3,bg='gold3',font=("Comic Sans MS", 12), padx=0, pady=0,width=30, height=15)
textwidget2.grid(row=0, column=0,columnspan=3)

b_back= Button(frame3, text='BACK', font=15, borderwidth=2,
                activebackground='chocolate3',command=click_back,bg='dark slate gray')
b_back.grid(row=2,column=1,padx=20,pady=20)

b_excel= Button(frame3, text='export to excel', font=15, borderwidth=2,
                activebackground='chocolate3',command=click_export,bg='dark slate gray')
b_excel.grid(row=2,column=2,padx=20,pady=20)

# frame4  select to compare
def click_back2():
    frame4.pack_forget()
    frame2.pack()

to_concat = []
def click_anadir():
    to_concat.append(p.get())

dfs_to_concat=[]
def click_concatenated():
    global zx,create_dfr, table
    frame4.pack_forget()
    frame5.pack()
    for zx, project in enumerate(to_concat):
        zx = apts[project].create_df()
        dfs_to_concat.append(zx)
    create_dfr = Concatenate_df(*dfs_to_concat)
    table = create_dfr.concatenate()
    textwidget5.insert(END, table)



l_select = Label(frame4,text='seleccione uno y add:',font=30,bg='gold3')
l_select.grid(row=0,column=0,columnspan=2,padx=20,pady=10)

b_anadir= Button(frame4,text='ADD',font=15,borderwidth=2,
                activebackground='chocolate3',command=click_anadir,bg='dark slate gray')
b_anadir.grid(row=10,column=1,padx=20,pady=20)

b_back2= Button(frame4,text='BACK',font=15,borderwidth=2,
                activebackground='chocolate3',command=click_back2,bg='dark slate gray')
b_back2.grid(row=10,column=2,padx=20,pady=20)


b_concatenated= Button(frame4,text='CREAR',font=15,borderwidth=2,
                activebackground='chocolate3',command=click_concatenated,bg='dark slate gray')
b_concatenated.grid(row=11,column=1,padx=20,pady=20)



# frame 5  compare selected

def click_back5():
    global  dfs_to_concat
    frame5.pack_forget()
    frame4.pack()
    dfs_to_concat = []

def click_export2():


    table.to_excel(r'C:\Users\Usuario\Desktop\comparasiones.xlsx')
    messagebox.showinfo('dd','done')



textwidget5 = Text(frame5,bg='gold3',font=("Comic Sans MS", 12), padx=0, pady=0,width=90, height=15)
textwidget5.grid(row=0, column=0,columnspan=3)

b_back= Button(frame5,text='BACK',font=15,borderwidth=2,
                activebackground='chocolate3',command=click_back5,bg='dark slate gray')
b_back.grid(row=2,column=1,padx=20,pady=20)


b_excel2= Button(frame5, text='export to excel', font=15, borderwidth=2,
                activebackground='chocolate3',command=click_export2,bg='dark slate gray')
b_excel2.grid(row=2,column=2,padx=20,pady=20)


root.mainloop()

