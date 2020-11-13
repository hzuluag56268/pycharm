import numpy as np
import  pandas as pd

initial = 1500000
interest = .043

time =    10
compounded = 1

#calculate compound interest
fv = initial  * (1 +  interest / compounded ) ** ( compounded * time)
print('your future of {} value is    {}'.format(initial,fv))

# #reverse compound interest, also for inflation
iv = initial  / (1 +  interest / compounded ) ** ( compounded * time)
print('your initial  value was    {}'.format(iv))

investment_1 = np.pv(rate=interest, nper=time, pmt=0, fv=initial) #calculate inflation
print("Investment 1 is worth " + str(round(-investment_1, 2)) + " in today's dollars")

print('............')
print('............')
home_value=159375000
loan= 127500000
monthly_payment= 1310401
anual_payment= monthly_payment*12
duration = 20
total_paid = anual_payment*duration

print('loan of {} with monthly payment of 1.310.401 for 20 years for total {}\n which is'
      ' {:.2f} times what you got'.format(loan,total_paid,total_paid/loan))
print('............')
print('............')
inflation_r = 0.045
values = np.repeat(anual_payment,20)

present_Value = np.npv(rate=inflation_r,values=values)
print('the discounted value of loan {} with inflation of {} is {}\n which is'
      ' {:.2f} times what you got'.format(loan,inflation_r,present_Value,loan/present_Value))


print('========')
print('========')
anual_rate= 0.115
month_rate = ((1+anual_rate)**(1/12)-1)
print('rate month',month_rate)
n_pmt = 12 * 20
a_loan = 127.5
p = -np.pmt(month_rate,n_pmt,a_loan)
print(p)


