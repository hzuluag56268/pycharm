'''1   The Time Value of Money


'''



..........Fundamental Financial Concepts






Growth and Rate of Return
# Calculate the future value of the investment and print it out
future_value = 100*(1+0.06)**(30)
print("Future Value of Investment: " + str(round(future_value, 2)))



Compound Interest

# Predefined variables
initial_investment = 100
growth_periods = 30
growth_rate = 0.06

# Calculate the value for the investment compounded once per year
compound_periods_1 = 1
investment_1 = initial_investment*(1 + growth_rate / compound_periods_1)**(compound_periods_1*growth_periods)
print("Investment 1: " + str(round(investment_1, 2)))

# Calculate the value for the investment compounded monthly
compound_periods_3 = 12
investment_3 = initial_investment*(1 + .6 / 12)**(12*30)  #the .6 is per year interest
#if you want to get the initial value, just multiply by discount factor which is
# initial_investment    /     (1 + .6 / 12)**(12*30) , divide not multiply 

print("Investment 3: " + str(round(investment_3, 2)))



Discount Factors and Depreciation

Calculate the final depreciated value of an initially
$10,000 car which declines in value by 3% per year for 10 years:

$10,000∗(1+−0.03)10=$7,374.24


# Calculate the future value
initial_investment = 100
growth_rate = -0.05
growth_periods = 10
future_value = initial_investment*(1 + growth_rate)**(growth_periods)
print("Future value: " + str(round(future_value, 2)))

# Calculate the discount factor
discount_factor = 1/((1 + growth_rate)**(growth_periods))
print("Discount factor: " + str(round(discount_factor, 2)))

# Derive the initial value of the investment
initial_investment_again = future_value*discount_factor
print("Initial value: " + str(round(initial_investment_again, 2)))



..............Present and Future Value


import numpy as np



# Calculate investment_1
investment_1 = np.fv(rate=0.08, nper=10, pmt=0, pv=-10000)
print("Investment 1 will yield a total of $" + str(round(investment_1, 2)) + " in 10 years")

# Calculate investment_2
investment_1_discounted = np.pv(rate=0.03, nper=10, pmt=0, fv=investment_1)
print("After adjusting for inflation, investment 1 is worth $" + str(round(-investment_1_discounted, 2)) + " in today's dollars")

<script.py> output:
    Investment 1 will yield a total of $21589.25 in 10 years
    After adjusting for inflation, investment 1 is worth $16064.43 in today's dollars




.........Net Present Value and Cash Flows
import numpy as np

# Calculate investment_1
investment_1 = np.pv(rate=0.03, nper=30, pmt=0, fv=100)
print("Investment 1 is worth $" + str(round(-investment_1, 2)) + " in today's dollars")

# Calculate investment_2
investment_2 = np.pv(rate=0.03, nper=50, pmt=0, fv=100)
print("Investment 2 is worth $" + str(round(-investment_2, 2)) + " in today's dollars")

# Calculate investment_3
investment_3 = np.pv(rate=0.03, nper=100, pmt=0, fv=100)
print("Investment 3 is worth $" + str(round(-investment_3, 2)) + " in today's dollars")
script.py> output:
    Investment 1 is worth $41.2 in today's dollars
    Investment 2 is worth $22.81 in today's dollars
    Investment 3 is worth $5.2 in today's dollars








'''2 Making Data-Driven Financial Decisions


'''


..........A Tale of Two Project Proposals
import numpy as np

# Calculate the internal rate of return for Project 1
irr_project1 = np.irr(cf_project1)
print("Project 1 IRR: " + str(round(100*irr_project1, 2)) + "%")

# Calculate the internal rate of return for Project 2
irr_project2 = np.irr(cf_project2)
print("Project 2 IRR: " + str(round(100*irr_project2, 2)) + "%")





...........The Weighted Average Cost of Capital


'''3 Simulating a Mortgage Loan


'''


................    1 Mortgage Basics

import numpy as np

# Set the value of the home you are looking to buy
home_value = 800000

# What percentage are you paying up-front?
down_payment_percent = 0.2

# Calculate the dollar value of the down payment
down_payment = home_value*down_payment_percent
print("Initial Down Payment: " + str(down_payment))

# Calculate the value of the mortgage loan required after the down payment
mortgage_loan = home_value- down_payment
print("Mortgage Loan: " + str(mortgage_loan))

import numpy as np

# Derive the equivalent monthly mortgage rate from the annual rate
mortgage_rate_periodic = ((1+0.0375)**(1/12))-1

# How many monthly payment periods will there be over 30 years?
mortgage_payment_periods = 12*30

# Calculate the monthly mortgage payment (multiply by -1 to keep it positive)
periodic_mortgage_payment = -1*np.pmt(mortgage_rate_periodic, mortgage_payment_periods, mortgage_loan)
print("Monthly Mortgage Payment: " + str(round(periodic_mortgage_payment, 2)))





..............Amortization, Interest and Principal


# Calculate the amount of the first loan payment that will go towards interest
initial_interest_payment = mortgage_loan*mortgage_rate_periodic
print("Initial Interest Payment: " + str(round(initial_interest_payment, 2)))

# Calculate the amount of the first loan payment that will go towards principal
initial_principal_payment = periodic_mortgage_payment - initial_interest_payment
print("Initial Principal Payment: " + str(round(initial_principal_payment, 2)))




# Loop through each mortgage payment period
for i in range(0, mortgage_payment_periods):

    # Handle the case for the first iteration
    if i == 0:
        previous_principal_remaining = mortgage_loan
    else:
        previous_principal_remaining = principal_remaining[i - 1]

    # Calculate the interest and principal payments
    interest_payment = round(previous_principal_remaining * mortgage_rate_periodic, 2)
    principal_payment = round(periodic_mortgage_payment - interest_payment, 2)

    # Catch the case where all principal is paid off in the final period
    if previous_principal_remaining - principal_payment < 0:
        principal_payment = previous_principal_remaining

    # Collect the principal remaining values in an array
    principal_remaining[i] = previous_principal_remaining - principal_payment

    # Print the payments for the first few periods
    print_payments(i, interest_payment, principal_payment, principal_remaining)




    # Loop through each mortgage payment period
    for i in range(0, mortgage_payment_periods):

        # Handle the case for the first iteration
        if i == 0:
            previous_principal_remaining = mortgage_loan
        else:
            previous_principal_remaining = principal_remaining[i - 1]

        # Calculate the interest based on the previous principal
        interest_payment = round(previous_principal_remaining * mortgage_rate_periodic, 2)
        principal_payment = round(periodic_mortgage_payment - interest_payment, 2)

        # Catch the case where all principal is paid off in the final period
        if previous_principal_remaining - principal_payment < 0:
            principal_payment = previous_principal_remaining

        # Collect the historical values
        interest_paid[i] = interest_payment
        principal_paid[i] = principal_payment
        principal_remaining[i] = previous_principal_remaining - principal_payment

    # Plot the interest vs principal
    plt.plot(interest_paid, color="red")
    plt.plot(principal_paid, color="blue")
    plt.legend(handles=[interest_plot, principal_plot], loc=2)
    plt.show()
    
    
    
    
..........Home Ownership, Home Prices and Recessions



import numpy as np

# Calculate the cumulative home equity (principal) over time
cumulative_home_equity = np.cumsum(principal_paid)

# Calculate the cumulative interest paid over time
cumulative_interest_paid = np.cumsum(interest_paid)

# Calculate your percentage home equity over time
cumulative_percent_owned = down_payment_percent + (cumulative_home_equity/ home_value)
print(cumulative_percent_owned)

# Plot the cumulative interest paid vs equity accumulated
plt.plot(cumulative_interest_paid, color='red')
plt.plot(cumulative_home_equity, color='blue')
plt.legend(handles=[interest_plot, principal_plot], loc=2)
plt.show()




import numpy as np

# Calculate the cumulative growth over time
cumulative_growth_forecast = np.cumprod(1+growth_array)


# Forecast the home value over time
home_value_forecast = home_value * cumulative_growth_forecast

# Forecast the home equity value owned over time
cumulative_home_value_owned = home_value_forecast*cumulative_percent_owned

# Plot the home value vs equity accumulated
plt.plot(home_value_forecast, color='red')
plt.plot(cumulative_home_value_owned, color='blue')
plt.legend(handles=[homevalue_plot, homeequity_plot], loc=2)
plt.show()


import numpy as np
import pandas as pd

# Cumulative drop in home value over time as a ratio
cumulative_decline_forecast = np.cumprod(1+decline_array)
print(cumulative_decline_forecast)
# Forecast the home value over time
home_value_forecast = home_value*cumulative_decline_forecast

# Find all periods where your mortgage is underwater
underwater = principal_remaining >home_value_forecast
pd.value_counts(underwater)

# Plot the home value vs principal remaining
plt.plot(home_value_forecast, color='red')
plt.plot(principal_remaining, color='blue')
plt.legend(handles=[homevalue_plot, principal_plot], loc=2)
plt.show()