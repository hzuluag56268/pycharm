from PARKINGAPP import *
placa = ['pl'+str(i) for i in range(5)]
marca = [ 'ma'+str(i) for i in range(5)]
modelo= [ 'mo'+str(i) for i in range(5)]
convenio= [1 for i in range(5)]
list_key_carro= ['placa', 'marca', 'modelo', 'convenio']

keys_cliente_table = ['cedula','primer_nombre','primer_apellido','apto','celular',
                      'carro_placa', 'conjunto_id']





cedul= [ 'ced'+str(i) for i in range(5)]
fn= [ 'pr'+str(i) for i in range(5)]
ln= [ 'se'+str(i) for i in range(5)]
apt= [ i for i in range(700,705)]
cel= [ 'cel'+str(i) for i in range(5)]
car_pla= ['pl'+str(i) for i in range(5)]
con_id= [1 for i in range(5)]

tuples_value_carro= [i for i in zip(placa, marca,
                                    modelo, convenio)]
tuples_value_cliente = [i for i in zip(cedul,fn,ln,apt,cel,car_pla,con_id)]

#for i in range(5):
#    add('carros', tuples_value_carro[i])

#for i in range(1,5):
    add('clientes', tuples_value_cliente[i])

#add('carros', tuples_value_carro[0])
#add('clientes', tuples_value_cliente[0])


#select_all_table('clientes')
