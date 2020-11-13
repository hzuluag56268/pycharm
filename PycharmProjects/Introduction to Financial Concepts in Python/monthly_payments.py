import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

home_value= 800000
down_payment_percent = 0.2
mortgage_loan = 640000.0
mortgage_rate_periodic = 0.003072541703255549
periodic_mortgage_payment = 2941.125363188976
n_cuotas = 12 * 30

array_interest, array_remaining, array_principal= np.empty(n_cuotas),np.empty(n_cuotas),np.empty(n_cuotas)
df_data= {'interest':array_interest,'principal':array_principal,'remaining':array_remaining}
remaining = mortgage_loan
for i in range(n_cuotas):

    interest_payment = remaining * mortgage_rate_periodic
    principal_payment = periodic_mortgage_payment - interest_payment
    remaining =  remaining- principal_payment

    array_remaining[i] = remaining
    array_interest[i] = interest_payment
    array_principal[i] = principal_payment
    if remaining < 0:
        array_remaining[i] = 0
        array_interest[i]= 0
        array_principal[i]= array_remaining[i-1]
    print( 'cuota: ',i+1, 'interest ',array_interest[i],'principal', array_principal[i],'remaining', array_remaining[i])

df = pd.DataFrame(df_data)


df['cumulative_interest'] = df['interest'].cumsum()
df['cumulative_principal'] = df['principal'].cumsum()
df['percentage_owned']     =  down_payment_percent + df['cumulative_principal'] / home_value


'''df[['cumulative_interest','cumulative_principal']].plot()
df[['percentage_owned']].plot()


plt.show()
'''

#calculate home equity'''

print('=============')
print('=============')
cumulative_growth = np.cumprod(1 + np.repeat(0.0025,df.shape[0]))

df['projhomval'] = home_value * cumulative_growth

df['equity'] = df['percentage_owned'] * df['projhomval']

df['r_equi']  =  (df['percentage_owned'] * home_value) + (df['projhomval'] - home_value)
print(df.head(2))

'''plt.plot(df['projhomval'] )
plt.plot(df['equity'], label='eq')
plt.plot(df['r_equi'] , label='re_eq')
plt.legend(loc='low')
plt.show()
'''


print('=============')
print('=============')

principal_remaining = mortgage_loan -  df['principal'].cumsum()
negative_cumulative_projection = np.cumprod(1 + np.repeat(-0.0045,df.shape[0]))
home_value_projection = home_value * negative_cumulative_projection
plt.plot(principal_remaining, label='principal')
plt.plot(home_value_projection, label='home')
plt.legend()
plt.show()

underwater = principal_remaining > home_value_projection
print(pd.value_counts(underwater))