import cost_calc2
import math
from random import *
import numpy as np

QP = cost_calc2.queryPlan
costs = cost_calc2.costDict
# print("QueryPlans=", QP)
# print("No of QP=", len(QP))
# print("costs = ", costs)

# Step 1: Fix the population size
population_size = len(costs)
print("population size=", population_size)

# Step 2: Fix the maximum number of iterations
max_iteration = 10

# solution for each generation and the fitness value of solution
costs_array = []
fitness_array = []
for i in range(population_size):
    # converting cost dictionary to array
    costs_array.append(costs[i + 1])
print("cost array =", costs_array)
decision_var = len(np.array(costs_array)[1, :])


# Step 3: Find out the fitness value for each member in the population
def fitness_func(member):
    result = 0
    for m in range(len(member)):
        result += math.sqrt(member[m])
    return result


# Step 4: selecting the best learner or teacher from the population based on minimum fitness value
def sel_teacher():
    # print("fitness value of each cost row= ", np.apply_along_axis(fitness_func, 1, costs_array))
    global fitness_array
    fitness_array = np.apply_along_axis(fitness_func, 1, costs_array)
    pos_teacher = np.where(fitness_array == np.amin(fitness_array))[0][0]
    # print("position of teacher = ", pos_teacher)
    return pos_teacher


# print("best QP, teacher =", QP[sel_teacher()])
# print("QAC, QLC of teacher= ", costs[sel_teacher()])
# print("QAC, QLC of teacher= ", costs_array[sel_teacher()-1])


# Step 5: determine mean of each cost of the population
def mean_column_costs():
    means = np.array(costs_array).mean(axis=0)
    print("numpy mean=", means)
    return means


# Step 6: Generation of new solution - Teacher phase of each student
def gen_new_soln():
    global costs_array
    global fitness_array
    X_best = np.array(costs_array[sel_teacher() - 1])
    X_mean = mean_column_costs()
    r = np.random.rand(decision_var)
    print("r=", r)
    Tf = round(random()) + 1
    print("tf=", Tf)
    for k in range(population_size):
        X = costs_array[k]
        X_new = X + r * (X_best - (Tf * X_mean))
        # Step 7: Before calculating fitness value for x_new, checking whether values in x_new are in between the lb
        # and ub condition of the fitness func
        X_new[(X_new < 0)] = 0
        X_new[(X_new > 2)] = 2
        # Step 8: Calculating the fitness value of the bounded solution
        fitness_x_new = fitness_func(X_new)
        # Step 9: Perform greedy selection to update the population
        fitness_teacher = fitness_func(X_best)
        if fitness_x_new < fitness_teacher:  # X_new is the best solution
            costs_array[k] = X_new
            fitness_array[k] = fitness_x_new



gen_new_soln()
