from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from sqlalchemy import create_engine, MetaData, Table, select, insert, delete, update, Column, String, Integer, ForeignKey, ForeignKeyConstraint
from sq_lite import *
#import psycopg2
#coonnect using psycopg2
'''con = psycopg2.connect(
    host='localhost',
    database= 'postgres',
    user= 'postgres',
    password= 'easy',
    port=1234
)

cursor = con.cursor()

con.close()'''
'''
url = "postgresql+psycopg2://harold:easy@localhost:1234/postgres"
engine = create_engine(url)
print(engine.table_names())

metadata = MetaData()
table_connect = Table('connect', metadata, autoload= True, autoload_with=engine)

print(repr(table_connect))
print(repr(metadata.tables['connect']))
print(table_connect.columns.keys())
print('----')

test = {'one': 4560, 'two': 800}

#insertz
def add(table, **data):
    with engine.connect() as conn:
        stm = insert(table)
        values = [data]
        result = conn.execute(stm, values)
        print('inserted : ', result.rowcount)
add(table_connect, **test)

def search_placa(table, table_columns_placa, placa):
    with engine.connect() as conn:
        conn = engine.connect()
        stmt = select([table]).where(table_columns_placa == placa)
        results = conn.execute(stmt).fetchall()
        print('placa : ', results)
search_placa(table_connect, table_connect.columns.one, 4560)

def update_value(table, table_colum_placa, num_placa, key_value):
    with engine.connect() as conn:
        stmt = update(table)
        stmt = stmt.where(table_colum_placa == num_placa)
        stmt = stmt.values(key_value)
        result_proxy = conn.execute(stmt)
        print('updated')
update_value(table_connect, table_connect.columns.one, 4560, {'one':1234})


def delete_placa(table,table_colum_placa,num_placa):
    with engine.connect() as conn:
        stm = delete(table).where(table_colum_placa == num_placa)
        proxy = conn.execute(stm)
        print('deleted ',proxy.rowcount)
delete_placa(table_connect, table_connect.columns.one, 1234)

'''



# Create the table in the database



def add(table, tuple_values):
    conn = sqlite3.connect('park.db')
    conn.execute("PRAGMA foreign_keys = 1")
    stm_select =" INSERT INTO {} values{}".format(table, tuple_values)
    c = conn.cursor()
    c.execute(stm_select)
    conn.commit()
    conn.close()
def drop_table(table):
    conn = sqlite3.connect('park.db')
    conn.execute("PRAGMA foreign_keys = 1")
    stm_select = " DELETE FROM {}".format(table)
    c = conn.cursor()
    c.execute(stm_select)
    conn.commit()
    conn.close()

def select_all_table(table):
    conn = sqlite3.connect('park.db')
    conn.execute("PRAGMA foreign_keys = 1")
    stm_select = "Select * from {}".format(table)
    c = conn.cursor()
    res = c.execute(stm_select).fetchall()
    conn.commit()
    conn.close()
    print(res)

#select_all_table('conjuntos')
#table,table2, table3, table_colum_placa, placa
def search_placa(plate):
    conn = sqlite3.connect('park.db')
    conn.execute("PRAGMA foreign_keys = 1")
    stm_select = '''
    SELECT cl.primer_nombre, cl.primer_apellido, cl.cedula, cl.celular, cl.apto,
    co.nombre, ca.placa, ca.marca, ca.modelo     
    FROM clientes AS cl
    INNER JOIN carros AS ca
    ON cl.carro_placa = ca.placa 
    INNER JOIN conjuntos AS co
    ON cl.conjunto_id = co.id 
    WHERE ca.placa = "%s"
    ''' %(plate)
    c = conn.cursor()
    res = c.execute(stm_select).fetchmany()
    conn.commit()
    conn.close()
    return res[0]

def update_table(table_name, attr_to_change,value,attr_where,value_where):
    conn = sqlite3.connect('park.db')
    conn.execute("PRAGMA foreign_keys = 1")
    stm_select = '''
    UPDATE %s
    SET "%s" = "%s"
    WHERE 
    "%s" = "%s"   
    '''%(table_name, attr_to_change,value,attr_where,value_where)
    c = conn.cursor()
    c.execute(stm_select)
    conn.commit()
    conn.close()
#update_table('CARROS','Placa', 'pl1', 'placa', 'pl10')
#select_all_table('clientes')

def delete_row(placa):
    conn = sqlite3.connect('park.db')
    conn.execute("PRAGMA foreign_keys = 1")
    stm_select = ''' 
    DELETE FROM carros
    WHERE placa = "%s"
    '''%(placa)
    c = conn.cursor()
    c.execute(stm_select)
    conn.commit()
    conn.close()
#delete_row()


select_all_table('carros')
select_all_table('clientes')
select_all_table('conjuntos')





root = Tk()
root.title('Park')
root.iconbitmap(r'lo.ico')


f_home = Frame(root)
f_home.pack()


def click_add():
   f_home.pack_forget()
   f_register.pack()

   print('add')
def click_gestion():
   f_home.pack_forget()
   f_gestion.pack()


b_add_car = Button(f_home, text='Agregar nuevo carro',command=click_add)
b_add_car.grid(row=0, column=0,sticky='WE',rowspan=2, columnspan=1,padx=0,pady=0)
b_gestion = Button(f_home, text='Gestionar Datos',command=click_gestion)
b_gestion.grid(row=0, column=1,sticky='WE',rowspan=2, columnspan=1,padx=0,pady=0)
img = ImageTk.PhotoImage(Image.open('car.jpg'))
l = Label(f_home, image=img)
l.grid(row=3, column=0,columnspan=2)






f_gestion = Frame(bg='#0f3057')



l_left_contain_gestion = Label(f_gestion, bg='#0f3057')
l_left_contain_gestion.grid(row=1, column=0,columnspan=1, rowspan=6)
e_placa1 = Entry(l_left_contain_gestion, bg='#16697a')
e_placa1.grid(row=1, column=0, pady=20)
cedula = ''
def click_consultar():
   global cedula
   placa = e_placa1.get()

   datos_tuple = search_placa(placa)
   print('adsf ',datos_tuple, placa)
   cedula = datos_tuple[2]

   searched_data = (' ').join([e.upper() + ' :' + ' ' * (20 - len(e)) + str(i) + '\n' for e, i in zip(attributes, datos_tuple)])
   l_output_datos.config(text=searched_data)
   b_actualizar.config(state=NORMAL)
   b_borrar.config(state=NORMAL)



b_placa = Button(l_left_contain_gestion, text='Consultar placa',command=click_consultar,bg='#0f3057',fg='#e7e7de')
b_placa.grid(row=0, column=0, padx=0, pady=20 )
attributes = ['Primer_Nombre', 'Primer_Apellido', 'Cedula', 'Celular', 'Apto', 'conjunto','placa', 'Marca', 'Modelo']
attribute_list = (' ').join([e.upper()+' :' +' '*(3-len(e)) + '\n' for e in attributes])
l_output_datos = Label(l_left_contain_gestion,text=attribute_list,anchor=W, justify=LEFT, bg='#0f3057',fg='#e7e7de')
l_output_datos.grid(row=2, column=0, padx=0, pady=0)

def click_actualizar():
   placa = e_placa1.get()
   if placa == '':
       messagebox.showinfo('','Ingrese placa')

   valor_actualizar = e_value_to_update.get()
   attr_actualizar = v_house.get()

   table_to_update = 'clientes'
   where_attr = 'cedula'
   where_attr_value= cedula

   options_of_carro = attributes[-3:]
   if attr_actualizar in options_of_carro:
       table_to_update = 'carros'
       where_attr = 'placa'
       where_attr_value= placa
   update_table(table_to_update, attr_actualizar, valor_actualizar,
                where_attr, where_attr_value)
   select_all_table('carros')
   v_house.set('Datos')
   e_value_to_update.delete(0,END)
   messagebox.showinfo('ACTUALIZADO', 'ACTUALIZACION'+' : \n++++++++  '+attr_actualizar +' : '+ valor_actualizar +'  +++++++++\nCOMPLETADO')
   print('actualizado', valor_actualizar, attr_actualizar)

l_right_up_contain_gestion = Label(f_gestion, bg='#00587a' )
l_right_up_contain_gestion.grid(row=1, column=1,columnspan=1, rowspan=4,padx=(0,30), pady=(20,0))
l_modificar = Label(l_right_up_contain_gestion, text='Hacer \n modificacion',bg='#00587a')
l_modificar.grid(row=0, column=0,rowspan=2,columnspan=2, padx=0, pady=0,sticky='we')
v_house = StringVar()
v_house.set('Datos')
o_houses = OptionMenu(l_right_up_contain_gestion, v_house, *attributes)
o_houses.config(bg='#00587a', activebackground='#00587a',fg='#e7e7de')
o_houses.grid(row=2, column=0, padx=5, pady=10,sticky='we')
e_value_to_update = Entry(l_right_up_contain_gestion, bg='#16697a')
e_value_to_update.grid(row=2, column=1)
l_ingrese_valor = Label(l_right_up_contain_gestion, text='^ ingrese nuevo ^ \nvalor', bg='#00587a')
l_ingrese_valor.grid(row=3, column=1,rowspan=1,columnspan=1, padx=0, pady=0)
b_actualizar = Button(l_right_up_contain_gestion,state=DISABLED, text='Actualizar base de datos',command=click_actualizar,bg='#00587a',fg='#e7e7de')
b_actualizar.grid(row=4, column=0, padx=0, pady=30,columnspan=2,sticky='we' )


def click_borrar():
   print('borrado')
   placa = e_placa1.get()
   if placa == '':
       messagebox.showinfo('', 'Ingrese placa')
   delete_row(placa)
   l_output_datos.config(text=attribute_list)
   b_borrar.config(state=DISABLED)
   messagebox.showinfo('','informacion Borrada')

def click_home():
   f_register.pack_forget()
   f_gestion.pack_forget()
   f_home.pack()
   print('home')
l_right_down_contain_gestion = Label(f_gestion ,bg='#0f3057')
l_right_down_contain_gestion.grid(row=5, column=1,columnspan=2, rowspan=2)
b_borrar = Button(l_right_down_contain_gestion,state=DISABLED, text='Borrar datos de placa\n consultada',command=click_borrar,activebackground='red',bg='#0f3057',fg='#e7e7de')
b_borrar.grid(row=1, column=0, padx=0, pady=30,columnspan=1,sticky='we' )
b_home = Button(l_right_down_contain_gestion, text=' home page ',command=click_home,activebackground='red',bg='#0f3057')
b_home.grid(row=2, column=1, padx=0, pady=0,columnspan=1,sticky='we' )








f_register = Frame(root)


l_convenio = Label(f_register, bg='#16697a')
l_convenio.grid(row=0, column=0,sticky='we', columnspan=4,padx=0, pady=0)
value_convenio = 0
def sel():
   global value_convenio
   value_convenio =var_convenio.get()
   selection = "You selected the option ", value_convenio
   print(selection )

var_convenio = IntVar()
r_c_convenio = Radiobutton(l_convenio, variable=var_convenio, command=sel, value=1, text='Con Convenio', bg='#16697a')
r_c_convenio.grid(row=0, column=2)
r_s_convenio = Radiobutton(l_convenio, variable=var_convenio, command=sel, value=0, text='Sin Convenio', bg='#16697a')
r_s_convenio.grid(row=0, column=4)

l_container1_f_register = Label(f_register, bg='#16697a')
l_container1_f_register.grid(row=1, column=0,columnspan=4, rowspan=2)

l_fn = Label(l_container1_f_register, text='Primer Nombre', bg='#16697a')
l_fn.grid(row=2, column=0, padx=0, pady=(20,10))
l_ln = Label(l_container1_f_register, text='Primer Apellido', bg='#16697a')
l_ln.grid(row=3, column=0)
l_id = Label(l_container1_f_register, text='Cedula', bg='#16697a')
l_id.grid(row=2, column=2, pady=(20,10), padx=(20,0))
l_phone = Label(l_container1_f_register, text='telefono', bg='#16697a')
l_phone.grid(row=3, column=2, padx=(20,0))
e_fn = Entry(l_container1_f_register, bg='#16697a')
e_fn.grid(row=2, column=1, pady=(20,10))
e_ln = Entry(l_container1_f_register, bg='#16697a')
e_ln.grid(row=3, column=1)
e_id = Entry(l_container1_f_register, bg='#16697a')
e_id.grid(row=2, column=3, pady=(20,10))
e_phone = Entry(l_container1_f_register, bg='#16697a')
e_phone.grid(row=3, column=3)




l_container_2_f_register = Label(f_register, bg='#db6400')
l_container_2_f_register.grid(row=3, column=0,columnspan=4, rowspan=2)

l_address = Label(l_container_2_f_register, text='Direccion', bg='#db6400')
l_address.grid(row=0, column=0, padx=0, pady=(20,10))
l_apt = Label(l_container_2_f_register, text='Apto', bg='#db6400')
l_apt.grid(row=0, column=1, pady=(20,10), padx=(17,17))
l_conjunto = Label(l_container_2_f_register, text='Conjunto', bg='#db6400')
l_conjunto.grid(row=0, column=3, pady=(20,10), padx=(0,0))

e_address = Entry(l_container_2_f_register, bg='#db6400')
e_address.grid(row=1, column=0, padx=0, pady=(0,25))
e_apt = Entry(l_container_2_f_register, bg='#db6400')
e_apt.grid(row=1, column=1, pady=(0,25), padx=(17,17))

v_conjunto = StringVar()
v_conjunto.set('brisas')
conjuntos_list = ['alto', 'bosques', 'brisas', 'dorado']
o_conjunto = OptionMenu(l_container_2_f_register, v_conjunto, *conjuntos_list)
o_conjunto.config(bg='#db6400', activebackground='#00587a',fg='#e7e7de')
o_conjunto.grid(row=1, column=4, padx=(0,0))




l_container_3_f_register = Label(f_register, bg='#16697a')
l_container_3_f_register.grid(row=5, column=0, sticky='we' ,columnspan=4, rowspan=2)

l_placa = Label(l_container_3_f_register, text='Placa', bg='#16697a')
l_placa.grid(row=2, column=0, padx=0, pady=(20,10))
l_marca = Label(l_container_3_f_register, text='Marca', bg='#16697a')
l_marca.grid(row=3, column=0)
l_modelo = Label(l_container_3_f_register, text='Modelo', bg='#16697a')
l_modelo.grid(row=2, column=2, pady=(20,10), padx=(60,0))
l_4digitos = Label(l_container_3_f_register, text='4 digitos', bg='#16697a')
l_4digitos.grid(row=3, column=2, padx=(60,0))


e_placa = Entry(l_container_3_f_register, bg='#16697a')
e_placa.grid(row=2, column=1, pady=(20,10))
e_marca = Entry(l_container_3_f_register, bg='#16697a')
e_marca.grid(row=3, column=1)
e_modelo = Entry(l_container_3_f_register, bg='#16697a')
e_modelo.grid(row=2, column=3, pady=(20,10))
e_4digitos = Entry(l_container_3_f_register, bg='#16697a')
e_4digitos.grid(row=3, column=3)

def click_save():
    messagebox.showinfo('', 'Tu carro ya fue registrado!:)')
    data_to_save = []
    print('saved')
    entries = [e_placa, e_marca, e_modelo, e_id,e_fn, e_ln,e_apt,e_phone, e_address,  e_4digitos]
    for e in entries:
        data_to_save.append(e.get())
        e.delete(0,END)

    if '' in data_to_save:
        print('please add all values')
    else:

        print(data_to_save)
    conjuto = 1
    if v_conjunto.get() == 'bosques':
        conjuto =2
    if v_conjunto.get() == 'brisas':
        conjuto = 3
    if v_conjunto.get() == 'dorado':
        conjuto = 4
    values_carro = (data_to_save[0], data_to_save[1], data_to_save[2], int(value_convenio))
    values_cliente = (data_to_save[3],data_to_save[4],data_to_save[5],data_to_save[6], data_to_save[7], data_to_save[0], conjuto)
    print(values_cliente)
    add('carros', values_carro)
    add('clientes',values_cliente)


    select_all_table('carros')

b_save = Button(f_register, text='SAVE', bg='#db6400',command=click_save)
b_save.grid(row=7, column=0, sticky='we' ,columnspan=4)


b_home = Button(f_register, text='HOME', bg='#db6400', command=click_home)
b_home.grid(row=8, column=0, sticky='we' ,columnspan=4)







root.mainloop()


