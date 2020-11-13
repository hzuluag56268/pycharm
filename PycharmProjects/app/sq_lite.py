import sqlite3

conn = sqlite3.connect('park.db')
conn.execute("PRAGMA foreign_keys = 1")
c = conn.cursor()

s2 = '''CREATE TABLE two(
four integer primary key,
five integer,
six integer

)'''
#c.execute(s2)

s = '''CREATE TABLE one(
one integer primary key,
two integer,
four integer NOT NULL,
FOREIGN KEY (four)
   REFERENCES two (four) 

)'''
#c.execute(s)
#conn.commit()
#c.execute("INSERT INTO one  VALUES('35','2','2000')")
#c.execute("INSERT INTO two  VALUES('206','101','104')")
#res = c.execute("SELECT * FROM one").fetchall()
#res2 = c.execute("SELECT * FROM two").fetchall()

stm_table_carros = '''CREATE TABLE IF NOT EXISTS carros (
    placa text primary key, 
    marca text NOT NULL,
    modelo text NOT NULL,
    convenio integer NOT NULL
)
'''

stm_table_conjuntos = '''CREATE TABLE IF NOT EXISTS conjuntos (
    id integer primary key,
    nombre text NOT NULL,
    direccion text NOT NULL
)
'''
stm_table_clientes = '''CREATE TABLE IF NOT EXISTS clientes (
cedula text primary key,
primer_nombre text NOT NULL,
primer_apellido text NOT NULL,
apto integer NOT NULL,
celular text NOT NULL,
carro_placa text NOT NULL,
conjunto_id integer NOT NULL,
FOREIGN KEY (carro_placa)
   REFERENCES carros (placa)
   ON DELETE CASCADE
   ON UPDATE CASCADE,
FOREIGN KEY (conjunto_id)
   REFERENCES conjuntos (id) 
   ON UPDATE CASCADE
   ON DELETE CASCADE

)
'''

c.execute(stm_table_carros)
c.execute(stm_table_conjuntos)
c.execute(stm_table_clientes)

'''
c.execute(" INSERT INTO conjuntos values(1,'alto','calle 9 14-43')")
c.execute(" INSERT INTO conjuntos values(2,'bosques','Ave 8 12-41')")
c.execute(" INSERT INTO conjuntos values(3,'brisas','Cll 11 16-9')")
c.execute(" INSERT INTO conjuntos values(4,'dorado','Cra 14 10-53')")
'''
conn.commit()
conn.close()