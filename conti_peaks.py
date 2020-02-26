import mlrose_hiive
import numpy as np
import datetime
import matplotlib.pyplot as plt
from util import generate_graph

"""
def our_fitness_func(state):
    global eval_count
    fitness = mlrose.FourPeaks(t_pct=0.15)
    eval_count += 1
    return fitness.evaluate(state)

fitness = mlrose.CustomFitness(our_fitness_func)
problem = mlrose.DiscreteOpt(length=50, fitness_fn=fitness, maximize=True)

eval_count = 0
best_state, best_fitness, curve = mlrose.thealgorithm(problem, ...)
print(eval_count)
"""

# global eval_count
# Count function evaluation counts
def cp_fitness_fn(state):
    global eval_count
    fitness = mlrose_hiive.ContinuousPeaks()
    eval_count += 1
    return fitness.evaluate(state)

def rhc(problem, iterations, random_seed, graph_file, graph_title):
    fitness = []
    fit_time = []
    fn_evals = []
    global eval_count
    for i in iterations:
        eval_count = 0
        start = datetime.datetime.now()
        best_state, best_fitness, _ = mlrose_hiive.random_hill_climb(problem,
                                   max_iters=i, random_state=random_seed)
        finish = datetime.datetime.now()
        fitness.append(best_fitness)
        fit_time.append((finish - start).total_seconds())
        fn_evals.append(eval_count)

    plt.plot(iterations, fitness, label="Fitness score")
    plt.legend(loc="best")
    plt.grid()
    generate_graph(graph_file + "rhc", graph_title + "Random Hill Climbing",
                   "Iterations", "Fitness")
    print('Best score achieved: ', max(fitness))
    index = fitness.index(max(fitness))
    print('Time taken to achieve that: ', fit_time[index])
    print('Function evaluations taken to achieve that: ', fn_evals[index])

def sa(problem, iterations, random_seed, graph_file, graph_title):
    decays = [0.001, 0.002, 0.003, 0.004, 0.005]
    best_score = []
    time_taken = []
    fn_evals_taken = []
    # fig1, ax1 = plt.subplots()
    # fig2, ax2 = plt.subplots()
    global eval_count
    for decay in decays:
        schedule = mlrose_hiive.ArithDecay(init_temp=1.0, decay=decay)
        fitness = []
        fit_time = []
        fn_evals = []
        for i in iterations:
            eval_count = 0
            start = datetime.datetime.now()
            # Solve using simulated annealing - attempt 1
            best_state, best_fitness, _ = mlrose_hiive.simulated_annealing(problem, schedule=schedule,
                                                                max_iters=i, random_state=random_seed)
            finish = datetime.datetime.now()
            fn_evals.append(eval_count)
            fitness.append(best_fitness)
            fit_time.append((finish - start).total_seconds())
            # print('iteration: ',i)
            # print('best_state:', best_state)
            # print('best_fitness: ', best_fitness)
        best_score.append(max(fitness))
        index = fitness.index(max(fitness))
        time_taken.append(fit_time[index])
        fn_evals_taken.append(fn_evals[index])
        # print('index: ', index)
        # print('time for that: ', fit_time[index])
        plt.plot(iterations, fitness, label="Cooling = " + str(decay))
        # ax2.plot(fn_evals, fitness, label="Cooling = " + str(decay))

    plt.legend(loc="best")
    plt.grid()
    generate_graph(graph_file + "sa_iter", graph_title + "Simulated Annealing", "Iterations", "Fitness")

    """
    ax2.legend(loc="best")
    ax2.grid()
    generate_graph("cp_sa_evals", "Continuous Peaks - Simulated Annealing", "Function evaluations", "Fitness")
    """
    # Decays best_score and time_taken
    plt.plot(decays, best_score)
    plt.grid()
    generate_graph(graph_file + "sa_decays", graph_title + "Simulated Annealing",
                   "Cooling Component", "Best Score Achieved")

    plt.plot(decays, time_taken)
    plt.grid()
    generate_graph(graph_file + "sa_decay_time", graph_title + "Simulated Annealing",
                   "Cooling Component", "Time taken to achieve that")

    plt.scatter(time_taken, best_score)
    for i, txt in enumerate(decays):
        plt.annotate(s="Cooling=" + str(txt), xy=(time_taken[i], best_score[i]))
    plt.grid()
    generate_graph(graph_file + "sa_scatter", graph_title + "Simulated Annealing",
                   "Time Taken", "Best Score achieved")

    print('decays: ', decays)
    print('Best scores reached: ', best_score)
    print('Time taken to do that: ', time_taken)
    print('Function evaluations taken: ', fn_evals_taken)

def ga(problem, iterations, random_seed, graph_file, graph_title):
    mutation_prob = [0.1, 0.2, 0.3, 0.4, 0.5]
    best_score = []
    time_taken = []
    fn_evals_taken = []
    global eval_count
    for m in mutation_prob:
        fitness = []
        fit_time = []
        fn_evals = []
        for i in iterations:
            eval_count = 0
            start = datetime.datetime.now()
            best_state, best_fitness, _ = mlrose_hiive.genetic_alg(problem, mutation_prob=m,
                                                                max_iters=i, random_state=None)
            finish = datetime.datetime.now()
            fitness.append(best_fitness)
            fit_time.append((finish - start).total_seconds())
            fn_evals.append(eval_count)
        # Find the best score achieved in that mutation prob
        best_score.append(max(fitness))
        index = fitness.index(max(fitness))
        # find the time that was taken to achieve that
        time_taken.append(fit_time[index])
        fn_evals_taken.append(fn_evals[index])
        plt.plot(iterations, fitness, label="Mutation = " + str(m))

    plt.legend(loc="best")
    plt.grid()
    generate_graph(graph_file + "ga", graph_title + "Genetic Algorithm", "Iterations", "Fitness")

    # Decays best_score and time_taken
    plt.plot(mutation_prob, best_score)
    plt.grid()
    generate_graph(graph_file + "ga_mut", graph_title + "Genetic Algorithm",
                   "Mutation Probability", "Best Score Achieved")

    """
    plt.plot(mutation_prob, time_taken)
    plt.grid()
    generate_graph("cp_sa_decay_time", "Continuous Peaks - Genetic Algorithm", "Mutation Probability",
                   "Time taken to achieve that")
    """

    plt.scatter(time_taken, best_score)
    for i, txt in enumerate(mutation_prob):
        plt.annotate(s="Mutation=" + str(txt), xy=(time_taken[i], best_score[i]))
    plt.grid()
    generate_graph(graph_file + "ga_scatter", graph_title + "Genetic Algorithm",
                   "Time Taken", "Best Score achieved")

    print('Mutation prob: ', mutation_prob)
    print('Best scores reached: ', best_score)
    print('Time taken to do that: ', time_taken)
    print('Function evaluations taken: ', fn_evals_taken)

def mimic(problem, iterations, random_seed, graph_file, graph_title):
    keep_pct = [0.25, 0.50, 0.75]
    best_score = []
    time_taken = []
    fn_evals_taken = []
    global eval_count
    for k in keep_pct:
        fitness = []
        fit_time = []
        fn_evals = []
        for i in iterations:
            eval_count = 0
            start = datetime.datetime.now()
            best_state, best_fitness, _ = mlrose_hiive.mimic(problem, keep_pct=k,
                                                            max_iters=i, random_state=None)
            finish = datetime.datetime.now()
            fitness.append(best_fitness)
            fit_time.append((finish - start).total_seconds())
            fn_evals.append(eval_count)
        # Find the best score achieved in that mutation prob
        best_score.append(max(fitness))
        index = fitness.index(max(fitness))
        # find the time that was taken to achieve that
        time_taken.append(fit_time[index])
        fn_evals_taken.append(fn_evals[index])
        plt.plot(iterations, fitness, label="keep_pct = " + str(k))

    plt.legend(loc="best", title='Proportion of samples kept')
    plt.grid()
    generate_graph(graph_file + "mimic", graph_title + "MIMIC: ", "Iterations", "Fitness")

    # Decays best_score and time_taken
    plt.plot(keep_pct, best_score)
    plt.grid()
    generate_graph(graph_file + "mimic_pct", graph_title + "MIMIC",
                   "Proportion of samples kept", "Best Score Achieved")

    """
    plt.plot(mutation_prob, time_taken)
    plt.grid()
    generate_graph("cp_sa_decay_time", "Continuous Peaks - Genetic Algorithm", "Mutation Probability",
                   "Time taken to achieve that")
    """

    plt.scatter(time_taken, best_score)
    for i, txt in enumerate(keep_pct):
        plt.annotate(s="keep_pct=" + str(txt), xy=(time_taken[i], best_score[i]))
    plt.grid()
    generate_graph(graph_file + "mimic_scatter", graph_title + "MIMIC",
                   "Time Taken", "Best Score achieved")

    print('Proportion of samples kept: ', keep_pct)
    print('Best scores reached: ', best_score)
    print('Time taken to do that: ', time_taken)
    print('Function evaluations taken: ', fn_evals_taken)

if __name__=="__main__":
    # Initialize fitness function object using Custom function
    fitness_fn = mlrose_hiive.CustomFitness(cp_fitness_fn)
    # fitness_fn = mlrose_hiive.ContinuousPeaks()
    # Define optimization problem object
    problem = mlrose_hiive.DiscreteOpt(length=50, fitness_fn=fitness_fn, max_val=2)
    max_iters = 1500
    iterations = range(0, max_iters, 50)
    random_seed = 1
    graph_file = 'cp/cp_'
    graph_title = 'Continuous Peaks - '
    # Random hill climbing
    rhc(problem, iterations, random_seed, graph_file, graph_title)
    # simulate annealing
    # sa(problem, iterations, random_seed, graph_file, graph_title)
    # Genetic Algorithm
    # ga(problem,iterations,random_seed, graph_file, graph_title)
    # MIMIC
    # mimic(problem, iterations, random_seed, graph_file, graph_title)