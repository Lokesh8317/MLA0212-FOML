import pandas as pd
from scipy.stats import norm

# Load the dataset
data = pd.read_csv('C:/Users/Pavan/Downloads/student-scores.csv')

# Separate data into passing and failing students
passing_students = data[data['mark'] >= 40]
failing_students = data[data['mark'] < 40]

# Estimate parameters for passing students
mean_marks_pass = passing_students['mark'].mean()
std_marks_pass = passing_students['mark'].std()
mean_total_pass = passing_students['total'].mean()
std_total_pass = passing_students['total'].std()

# Estimate parameters for failing students
mean_marks_fail = failing_students['mark'].mean()
std_marks_fail = failing_students['mark'].std()
mean_total_fail = failing_students['total'].mean()
std_total_fail = failing_students['total'].std()

# Prior probabilities
p_pass = len(passing_students) / len(data)
p_fail = 1 - p_pass

# Function to predict pass or fail using Bayesian decision theory
def bayesian_decision(marks, total):
    # Likelihood of passing
    likelihood_pass_marks = norm.pdf(mark, mean_marks_pass, std_marks_pass)
    likelihood_pass_total = norm.pdf(total, mean_total_pass, std_total_pass)
    likelihood_pass = likelihood_pass_marks * likelihood_pass_total

    # Likelihood of failing
    likelihood_fail_marks = norm.pdf(mark, mean_marks_fail, std_marks_fail)
    likelihood_fail_total = norm.pdf(total, mean_total_fail, std_total_fail)
    likelihood_fail = likelihood_fail_marks * likelihood_fail_total

    # Posterior probabilities
    posterior_pass = likelihood_pass * p_pass
    posterior_fail = likelihood_fail * p_fail

    return "Pass" if posterior_pass > posterior_fail else "Fail"

# Example
mark = 30
total = 44
decision = bayesian_decision(mark, total)
print(f"Example student with marks {mark} and total {total} is predicted to: {decision}")
