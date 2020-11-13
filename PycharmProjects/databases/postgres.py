from sqlalchemy import create_engine, MetaData, Table, select, insert, delete, update, Column, String, Integer, ForeignKey
from tkinter import *

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


engine = create_engine('sqlite:///parking.sqlite')

# Initialize MetaData: metadata
metadata = MetaData()
carros = Table('carros', metadata,
               Column('placa', String(10), primary_key=True),
               Column('marca', String(10), nullable=FALSE),
               Column('modelo',String(10), nullable=FALSE),
               Column('convenio', Integer(), nullable=FALSE)
               )
conjuntos = Table('conjuntos', metadata,
               Column('id', Integer(), primary_key=True),
               Column('nombre', String(20), nullable=FALSE),
               Column('direccion',String(30),nullable=FALSE)
               )

clientes = Table('clientes', metadata,
               Column('cedula', String(20), primary_key=True),
               Column('primer_nombre', String(20), nullable=FALSE),
               Column('primer_apellido',String(20),nullable=FALSE),
               Column('apto',Integer(),nullable=FALSE),
               Column('celular',String(20),nullable=FALSE),
               Column('carro_placa',String(10),ForeignKey("carros.placa"),nullable=FALSE),
               Column('conjunto_id',Integer(),ForeignKey("conjuntos.id"),nullable=FALSE)
               )



# Create the table in the database
metadata.create_all(engine)
print(engine.table_names())
def add(table, **data):
    with engine.connect() as conn:
        stm = insert(table)
        values = [data]
        result = conn.execute(stm, values)
        print('inserted : ', result.rowcount)
add(clientes, **{'cedula':'ertr','primer_nombre':'sfg', 'primer_apellido':'sfg', 'apto':30, 'celular':'sfg', 'carro_placa':'sfg', 'conjunto_id':1012})
#add(carros, **{'placa':'rtr', 'marca':'adsf','modelo':'asdf', 'convenio':'1'})
#add(conjuntos, **{'id':1012,'nombre':'adfadf', 'direccion':'asdfaf'})
def search_placa(table):
    with engine.connect() as conn:
        conn = engine.connect()
        stmt = select([table])
        results = conn.execute(stmt).fetchall()
        print('placa : ', results)
#search_placa(carros)
#search_placa(conjuntos)
search_placa(clientes)
'''def add(table, **data):
    with engine.connect() as conn:
        stm = insert(table)
        values = [data]
        result = conn.execute(stm, values)
        print('inserted : ', result.rowcount)
add(census, **{'age':34, 'pop2008':123})

def search(table):
    with engine.connect() as conn:
        stm = select([table])
        result = conn.execute(stm).fetchall()
        return result
res = search(census)


root = Tk()
t = Text(root)
t.pack()
t.insert(END, res)

root.mainloop()'''