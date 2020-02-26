import numpy as np
import matplotlib.pyplot as plt

from util import generate_graph

def one_max():
  algorithms = ['RHC', 'SA', 'GA', 'MIMIC']
  best_score_om = [46, 44, 50, 50]
  time_taken_om = [0.00773, 0.006309, 0.554985, 19.869137]
  fn_evals_om = [88, 214, 6039, 3221]
  x = np.arange(4)
  colors = ['coral', 'orange', 'mediumseagreen', 'cornflowerblue']
  # Best Score achieved
  plt.bar(x, height= best_score_om, color=colors)
  plt.xticks(x, algorithms)
  generate_graph("one_max_score", "One Max - Best Scores", "Algorithms", "Best Score Achieved")

  # Time taken to achieve that
  plt.bar(x, height= time_taken_om, color=colors)
  plt.xticks(x, algorithms)
  generate_graph("one_max_time", "One Max - Running Time", "Algorithms", "Time taken to achieve that")

  # Time taken to achieve that
  plt.bar(x, height= fn_evals_om, color=colors)
  plt.xticks(x, algorithms)
  generate_graph("one_max_evals", "One Max - Function evaluations", "Algorithms", "Function evaluations taken")

def cp():
  algorithms = ['RHC', 'SA', 'GA', 'MIMIC']
  best_score_om = [56, 84, 94, 85]
  time_taken_om = [0.002085, 0.048171, 1.746986, 43.326225]
  fn_evals_om = [13, 819, 12880, 6846]
  x = np.arange(4)
  colors = ['coral', 'orange', 'mediumseagreen', 'cornflowerblue']
  # Best Score achieved
  plt.bar(x, height= best_score_om, color=colors)
  plt.xticks(x, algorithms)
  generate_graph("cp_score", "Continuous Peaks - Best Scores", "Algorithms", "Best Score Achieved")

  # Time taken to achieve that
  plt.bar(x, height= time_taken_om, color=colors)
  plt.xticks(x, algorithms)
  generate_graph("cp_time", "Continuous Peaks - Running Time", "Algorithms", "Time taken to achieve that")

  # Time taken to achieve that
  plt.bar(x, height= fn_evals_om, color=colors)
  plt.xticks(x, algorithms)
  generate_graph("cp_evals", "Continuous Peaks - Function evaluations", "Algorithms", "Function evaluations taken")

def ks():
  algorithms = ['RHC', 'SA', 'GA', 'MIMIC']
  best_score_om = [41, 45, 50, 50]
  time_taken_om = [0.002853, 0.007676, 0.287459, 0.608017]
  fn_evals_om = [18, 28, 2615, 2413]
  x = np.arange(4)
  colors = ['coral', 'orange', 'mediumseagreen', 'cornflowerblue']
  # Best Score achieved
  plt.bar(x, height= best_score_om, color=colors)
  plt.xticks(x, algorithms)
  generate_graph("ks_score", "Knapsack - Best Scores", "Algorithms", "Best Score Achieved")

  # Time taken to achieve that
  plt.bar(x, height= time_taken_om, color=colors)
  plt.xticks(x, algorithms)
  generate_graph("ks_time", "Knapsack - Running Time", "Algorithms", "Time taken to achieve that")

  # Time taken to achieve that
  plt.bar(x, height= fn_evals_om, color=colors)
  plt.xticks(x, algorithms)
  generate_graph("ks_evals", "Knapsack - Function evaluations", "Algorithms", "Function evaluations taken")

if __name__=="__main__":
    one_max()
    cp()
    ks()