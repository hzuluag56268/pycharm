'''3 Making Model Predictions


'''


..........Modeling Real Data

# Import LinearRegression class, build a model, fit to the data
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(years, levels)

# Use model to make a prediction for one year, 2100
future_year = 2100
future_level = model.predict(future_year)
print("Prediction: year = {}, level = {:.02f}".format(future_year, future_level[0,0]))




# Fit the model, based on the form of the formula
model_fit = ols(formula="velocities ~ distances", data=df).fit()

# Extract the model parameters and associated "errors" or uncertainties
a0 = model_fit.params['Intercept']
a1 = model_fit.params['distances']
e0 = model_fit.bse['Intercept']
e1 = model_fit.bse['distances']




...........The Limits of Prediction





# Build the model and compute the residuals "model - data"
y_model = model_fit_and_predict(x_data, y_data)
residuals = y_model - y_data

# Compute the RSS, MSE, and RMSE and print the results
RSS = np.sum(np.square(residuals))
MSE = RSS/len(residuals)
RMSE = np.sqrt(MSE)
print('RMSE = {:0.2f}, MSE = {:0.2f}, RSS = {:0.2f}'.format(RMSE, MSE, RSS))



# Compute the residuals and the deviations
residuals = y_model - y_data
deviations = np.mean(y_data) - y_data

# Compute the variance of the residuals and deviations
var_residuals = np.mean(np.square(residuals))
var_deviations = np.mean(np.square(deviations))

# Compute r_squared as 1 - the ratio of RSS/Variance
r_squared = 1 - (var_residuals /var_deviations)
print('R-squared is {:0.2f}'.format(r_squared))

.......Standard Error





'''4  Estimating Model Parameters


'''


.........Inferential Statistics Concepts




# Initialize two arrays of zeros to be used as containers
means = np.zeros(num_samples)
stdevs = np.zeros(num_samples)

# For each iteration, compute and store the sample mean and sample stdev
for ns in range(num_samples):
    sample = np.random.choice(population, num_pts)
    means[ns] = sample.mean()
    stdevs[ns] = sample.std()

# Compute and print the mean() and std() for the sample statistic distributions
print("Means:  center={:>6.2f}, spread={:>6.2f}".format(means.mean(), means.std()))
print("Stdevs: center={:>6.2f}, spread={:>6.2f}".format(stdevs.mean(), stdevs.std()))




.........Model Estimation and Likelihood






# Compute the mean and standard deviation of the sample_distances
sample_mean = np.mean(sample_distances)
sample_stdev = np.std(sample_distances)

# Use the sample mean and stdev as estimates of the population model parameters mu and sigma
population_model = gaussian_model(sample_distances, mu=sample_mean, sigma=sample_stdev)

# Plot the model and data to see how they compare
fig = plot_data_and_model(sample_distances, population_model)






# Compute sample mean and stdev, for use as model parameter value guesses
mu_guess = np.mean(sample_distances)
sigma_guess = np.std(sample_distances)

# For each sample distance, compute the probability modeled by the parameter guesses
probs = np.zeros(len(sample_distances))
for n, distance in enumerate(sample_distances):
    probs[n] = gaussian_model(distance, mu=mu_guess, sigma=sigma_guess)

# Compute and print the log-likelihood as the sum() of the log() of the probabilities
loglikelihood = np.sum(np.log(probs))
print('For guesses mu={:0.2f} and sigma={:0.2f}, the loglikelihood={:0.2f}'.format(mu_guess, sigma_guess, loglikelihood))





# Create an array of mu guesses, centered on sample_mean, spread out +/- by sample_stdev
low_guess = sample_mean - 2*sample_stdev
high_guess = sample_mean + 2*sample_stdev
mu_guesses = np.linspace(low_guess, high_guess, 101)

# Compute the loglikelihood for each model created from each guess value
loglikelihoods = np.zeros(len(mu_guesses))
for n, mu_guess in enumerate(mu_guesses):
    loglikelihoods[n] = compute_loglikelihood(sample_distances, mu=mu_guess, sigma=sample_stdev)

# Find the best guess by using logical indexing, the print and plot the result
best_mu = mu_guesses[loglikelihoods==np.max(loglikelihoods)]
print('Maximum loglikelihood found for best mu guess={}'.format(best_mu))
fig = plot_loglikelihoods(mu_guesses, loglikelihoods)





.......Model Uncertainty and Sample Distributions

# Use the sample_data as a model for the population
population_model = sample_data

# Resample the population_model 100 times, computing the mean each sample
for nr in range(num_resamples):
    bootstrap_sample = np.random.choice(population_model, size=resample_size, replace=True)
    bootstrap_means[nr] = np.mean(bootstrap_sample)

# Compute and print the mean, stdev of the resample distribution of means
distribution_mean = np.mean(bootstrap_means)
standard_error = np.std(bootstrap_means)
print('Bootstrap Distribution: center={:0.1f}, spread={:0.1f}'.format(distribution_mean, standard_error))

# Plot the bootstrap resample distribution of means
fig = plot_data_hist(bootstrap_means)




# Resample each preloaded population, and compute speed distribution
population_inds = np.arange(0, 99, dtype=int)
for nr in range(num_resamples):
    sample_inds = np.random.choice(population_inds, size=100, replace=True)
    sample_inds.sort()
    sample_distances = distances[sample_inds]
    sample_times = times[sample_inds]
    a0, a1 = least_squares(sample_times, sample_distances)
    resample_speeds[nr] = a1

# Compute effect size and confidence interval, and print
speed_estimate = np.mean(resample_speeds)
ci_90 = np.percentile(resample_speeds, [5, 95])
print('Speed Estimate = {:0.2f}, 90% Confidence Interval: {:0.2f}, {:0.2f} '.format(speed_estimate, ci_90[0], ci_90[1]))



..............Model Errors and Randomness




# From the unshuffled groups, compute the test statistic distribution
resample_short = np.random.choice(group_duration_short, size=500, replace=True)
resample_long = np.random.choice(group_duration_long, size=500, replace=True)
test_statistic_unshuffled = resample_long - resample_short

# Shuffle two populations, cut in half, and recompute the test statistic
shuffled_half1, shuffled_half2 = shuffle_and_split(group_duration_short, group_duration_long)
resample_half1 = np.random.choice(shuffled_half1, size=500, replace=True)
resample_half2 = np.random.choice(shuffled_half2, size=500, replace=True)
test_statistic_shuffled = resample_half2 - resample_half1

# Plot both the unshuffled and shuffled results and compare
fig = plot_test_statistic(test_statistic_unshuffled, label='Unshuffled')
fig = plot_test_statistic(test_statistic_shuffled, label='Shuffled')

















'''Statistical Simulation in Python


'''





...'''Basics of randomness & simulation'''




.............Introduction to random variables


# Shuffle the deck
np.random.shuffle(deck_of_cards)

# Print out the top three cards
card_choices_after_shuffle = deck_of_cards[:3]
print(card_choices_after_shuffle)



..........Simulation basics



# Initialize model parameters & simulate dice throw
die, probabilities, num_dice = [1,2,3,4,5,6], [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], 2
sims, wins = 100, 0

for i in range(sims):
    outcomes = np.random.choice(die,num_dice,p=probabilities)
    # Increment `wins` by 1 if the dice show same number
    if outcomes[0] == outcomes[-1]:
        wins = wins + 1

print("In {} games, you win {} times".format(sims, wins))



..........Using simulation for decision-making


    # Initialize size and simulate outcome
    lottery_ticket_cost, num_tickets, grand_prize = 10, 1000, 1000000
    chance_of_winning = 1 / num_tickets
    size = 2000
    payoffs = [- lottery_ticket_cost,        grand_prize - lottery_ticket_cost]

    probs = [  1 - chance_of_winning,        chance_of_winning]

    outcomes = np.random.choice(a=payoffs, size=size, p=probs, replace=True)

    # Mean of outcomes.
    answer = outcomes.mean()
    print("Average payoff from {} simulations = {}".format(size, answer))





    # Initialize simulations and cost of ticket
    sims, lottery_ticket_cost = 3000, 0

    # Use a while loop to increment `lottery_ticket_cost` till average value of outcomes falls below zero
    while 1:
        outcomes = np.random.choice([-lottery_ticket_cost, grand_prize - lottery_ticket_cost],
                                    size=sims, p=[1 - chance_of_winning, chance_of_winning], replace=True)
        if outcomes.mean() < 0:
            break
        else:
            lottery_ticket_cost += 1
    answer = lottery_ticket_cost - 1

    print("The highest price at which it makes sense to buy the ticket is {}".format(answer))









    '''Probability & data generation process

    '''



    .......Probability basics


P(A∪B)=P(A)+P(B)−P(A∩B)


# Shuffle deck & count card occurrences in the hand
n_sims, two_kind = 10000, 0
for i in range(n_sims):
    np.random.shuffle(deck_of_cards)
    hand, cards_in_hand = deck_of_cards[0:5], {}
    for card in hand:
        # Use .get() method on cards_in_hand
        cards_in_hand[card[1]] = cards_in_hand.get(card[1], 0) + 1

    # Condition for getting at least 2 of a kind
    highest_card = max(cards_in_hand.values())
    if highest_card >= 2:
        two_kind += 1

print("Probability of seeing at least two of a kind = {} ".format(two_kind / n_sims))




# Pre-set constant variables
deck, sims, coincidences = np.arange(1, 20), 10000, 0

for _ in range(sims):
    # Draw all the cards without replacement to simulate one game
    draw = np.random.choice(deck, size=len(deck), replace=False)
    # Check if there are any coincidences
    coincidence = (draw == list(np.arange(1, 20))).any()
    if coincidence == True:
        coincidences += 1
# Calculate probability of winning
prob_of_winning =1- (coincidences/ sims)
print("Probability of winning = {}".format(prob_of_winning))




......More probability concepts




# Initialize success, sims and urn
success, sims = 0, 5000
urn = ['w','w','w','w','w','w','w','b','b','b','b','b','b']

for _ in range(sims):
    # Draw 4 balls without replacement
    draw = np.random.choice(urn, replace=False, size=4)
    # Count the number of successes
    if draw[0] == 'w' and draw[2] == 'w' and draw[1] == 'b' and draw[3] == 'b' :
        success +=1

print("Probability of success = {}".format(success/sims))




# Draw a sample of birthdays & check if each birthday is unique
days = np.arange(1,366)
people = 2

def birthday_sim(people):
    sims, unique_birthdays = 2000, 0
    for _ in range(sims):
        draw = np.random.choice(days, size=people, replace=True)
        if len(draw) == len(set(draw)):
            unique_birthdays += 1
    out = 1 - unique_birthdays / sims
    return out


# Break out of the loop if probability greater than 0.5
while (people > 0):
    prop_bds = birthday_sim(people)
    if prop_bds > 0.5:
        break
    people += 1

print("With {} people, there's a 50% chance that two share a birthday.".format(people))





# Shuffle deck & count card occurrences in the hand
n_sims, full_house, deck_of_cards = 50000, 0, deck.copy()
for i in range(n_sims):
    np.random.shuffle(deck_of_cards)
    hand, cards_in_hand = deck_of_cards[0:5], {}
    for card in hand:
        # Use .get() method to count occurrences of each card
        cards_in_hand[card[1]] = cards_in_hand.get(card[1], 0) + 1

    # Condition for getting full house
    condition = (max(cards_in_hand.values()) == 3) & (min(cards_in_hand.values()) == 2)
    if condition == True:
        full_house += 1
print("Probability of seeing a full house = {}".format(full_house / n_sims))



........Data generating process



sims, outcomes, p_rain, p_pass = 1000, [], 0.40, {'sun':0.9, 'rain':0.3}

def test_outcome(p_rain):
    # Simulate whether it will rain or not
    weather = np.random.choice(['rain', 'sun'], p=[p_rain, 1-p_rain])
    # Simulate and return whether you will pass or fail
    return np.random.choice(['pass', 'fail'], p=[p_pass[weather], 1-p_pass[weather]])

for _ in range(sims):
    outcomes.append(test_outcome(p_rain))

# Calculate fraction of outcomes where you pass
pass_outcomes_frac = outcomes.count('pass')/ len(outcomes)
print("Probability of Passing the driving test = {}".format(pass_outcomes_frac))




outcomes, sims, probs = [], 1000, p

for _ in range(sims):
    # Simulate elections in the 50 states
    election = np.random.binomial(p=probs, n=1)
    # Get average of Red wins and add to `outcomes`
    outcomes.append(election.mean())

# Calculate probability of Red winning in less than 45% of the states
prob_red_wins = sum([(x < 0.45) for x in outcomes]) / len(outcomes)
print("Probability of Red winning in less than 45% of the states = {}".format(prob_red_wins))





# Simulate steps & choose prob
for _ in range(sims):
    w = []
    for i in range(days):
        lam = np.random.choice([5000, 15000], p=[0.6, 0.4], size=1)
        steps = np.random.poisson(lam)
        if steps > 10000:
            prob = [0.2, 0.8]
        elif steps < 8000:

            prob = [0.8, 0.2]
        else:
            prob = [0.5, 0.5]
        w.append(np.random.choice([1, -1], p=prob))
    outcomes.append(sum(w))

# Calculate fraction of outcomes where there was a weight loss
weight_loss_outcomes_frac = sum([x < 0 for x in outcomes]) / len(outcomes)
print("Probability of Weight Loss = {}".format(weight_loss_outcomes_frac))





.............eCommerce Ad Simulation






# Initialize click-through rate and signup rate dictionaries
ct_rate = {'low':0.01, 'high':np.random.uniform(low=0.01, high=1.2*0.01)}
su_rate = {'low':0.2, 'high':np.random.uniform(low=0.2, high=1.2*0.2)}

def get_signups(cost, ct_rate, su_rate, sims):
    lam = np.random.normal(loc=100000, scale=2000, size=sims)
    # Simulate impressions(poisson), clicks(binomial) and signups(binomial)
    impressions = np.random.poisson(lam)
    clicks = np.random.binomial(n=impressions,p=ct_rate[cost])
    signups = np.random.binomial(n=clicks,p=su_rate[cost])
    print(cost)
    return signups

print("Simulated Signups = {}".format(get_signups('high', ct_rate, su_rate, 1)))







'''CASE STUDY
'''



.......