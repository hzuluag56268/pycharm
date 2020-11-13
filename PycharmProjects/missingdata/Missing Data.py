'''
The Problem With Missing Data\\\



'''

........Why deal with missing data?

........Handling missing values

college = pd.read_csv('college.csv', na_values='.')



# Store all rows of column 'BMI' which are equal to 0
zero_bmi = diabetes.BMI[diabetes.BMI == 0]
print(zero_bmi)

# Set the 0 values of column 'BMI' to np.nan
diabetes.BMI[diabetes.BMI == 0] = np.nan

# Print the 'NaN' values in the column BMI
print(diabetes.BMI[np.isnan(diabetes.BMI)])



.............Analyze the amount of missingness


# Load the airquality dataset
airquality = pd.read_csv('air-quality.csv', parse_dates=['Date'], index_col='Date')



# Create a nullity DataFrame airquality_nullity
airquality_nullity = airquality.isnull()
print(airquality_nullity.head())

# Calculate total of missing values
missing_values_sum = airquality_nullity.sum()
print('Total Missing Values:\n', missing_values_sum)

# Calculate percentage of missing values
missing_values_percent = airquality_nullity.mean() * 100
print('Percentage of Missing Values:\n', missing_values_percent)





# Import missingno as msno
import missingno as msno

# Plot amount of missingness
msno.bar(airquality)




# Import missingno as msno
import missingno as msno

# Plot the sliced nullity matrix of airquality with frequency 'M'
msno.matrix(airquality.loc['May-1976':'Jul-1976'], freq='M')



'''Does Missingness Have A Pattern?



'''


......Is the data missing at random?


...........Finding patterns in missing data


# Import missingno
import missingno as msno

# Plot missingness heatmap of diabetes
msno.heatmap(diabetes)

# Plot missingness dendrogram of diabetes
msno.dendrogram(diabetes)

# Show plot
plt.show()

...




# Visualize the missingness of diabetes prior to dropping missing values
msno.matrix(diabetes)

# Print the number of missing values in Glucose
print(diabetes['Glucose'].isnull().sum())

# Drop rows where 'Glucose' has a missing value
diabetes.dropna(subset=['Glucose'], how='any', inplace=True)

# Visualize the missingness of diabetes after dropping missing values
msno.matrix(diabetes)

display("/usr/local/share/datasets/glucose_dropped.png")




''' imputation



'''



.........Mean, median & mode imputations

SimpleImputer() from sklearn.impute


# Make a copy of diabetes
diabetes_median = diabetes.copy(deep=True)

# Create median imputer object

median_imputer = SimpleImputer(strategy='median')
mean_imputer = SimpleImputer(strategy='mean')
constant_imputer = SimpleImputer(strategy="constant", fill_value=0)
mode_imputer = SimpleImputer(strategy='most_frequent')




# Impute median values in the DataFrame diabetes_median
diabetes_median.iloc[:, :] = median_imputer.fit_transform(diabetes_median)



.....Imputing time-series data


# Fill NaNs using backward fill
airquality.fillna(method='bfill', inplace=True)


# Interpolate the NaNs with nearest value
airquality.interpolate(method='nearest', inplace=True)


# Interpolate the NaNs with nearest value
airquality.interpolate(method='nearest', inplace=True)




'''Advanced Imputation Techniques


'''




...........Imputing using fancyimpute
