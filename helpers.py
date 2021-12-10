import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import random
import math
from joblib import Parallel, delayed

class AnnealTVS():
    def __init__(self, dataframe, num_sim=10, K = 2, stopK = 0.1, alpha = 0.99, ittol = 10000, elementary = "triangle", verbose=False):
        self.df = dataframe
        self.K = K
        self.startK = K
        self.stopK = stopK
        self.alpha = alpha
        self.ittol = ittol
        self.elementary = elementary
        self.verbose = verbose
        self.num_sim = num_sim

        # start with a simple solution
        self.solution = self.nearest_neighbours()


    def run_sim(self):
        results = []
        for i in range(self.num_sim):
            results.append(self.sim_anneal())
        return results


    def sim_anneal(self):
        assert self.alpha < 1, "choose a smaller alpha"
        all_dist = []
        old_dist = self.total_distance()
        i = 0
        if self.verbose:
            lowered = 0
            raised = 0
        while self.K > self.stopK and i < self.ittol:
            # elementary edit, triangle swap for now
            if self.elementary =="triangle":
                new_solution = self.triangle_swap()
            elif self.elementary == "swap":
                new_solution = self.swap()
            elif self.elementary == "shuffle":
                new_solution = self.shuffle()
            else:
                raise ValueError
                
            new_dist = self.total_distance(new_solution)
            # decide whether to accept the new solution
            if new_dist < old_dist:
                if self.verbose:
                    lowered += 1
                self.solution = new_solution
                old_dist = self.total_distance()
                all_dist.append(self.total_distance())
            else:
                # accept with probability depending on temperature
                rand = np.random.random()
                prob = math.exp(-abs(new_dist - old_dist) / self.K )
                if rand < prob:
                    if self.verbose:
                        raised += 1
                    self.solution = new_solution
                    old_dist = self.total_distance()
                    all_dist.append(self.total_distance())
            
            # decrease temperature
            i += 1
            self.K *= self.alpha 
        
        self.K = self.startK
        if self.verbose:
            print(f"times lowered: {lowered}\ntimes raised:{raised}")
        return self.solution, all_dist


    def nearest_neighbours(self):
        free_nodes = []
        for row in range(self.df.shape[0]):
            free_nodes.append(self.df.iloc[row].name)

        sol = []
        cur_node = self.df.loc[random.choice(free_nodes)]
        sol.append(cur_node.name)
        free_nodes.remove(cur_node.name)

        while free_nodes:
            first = True
            for node in free_nodes:
                next_node = self.df.loc[node]
                new_dist = self.get_distance(cur_node["x"], cur_node["y"], next_node["x"], next_node["y"])
                if first:
                    closest = node
                    old_dist = new_dist
                    first = False
                else: 
                    if new_dist < old_dist:
                        closest = node
                        old_dist = new_dist
            sol.append(closest)
            cur_node = self.df.loc[closest]
            free_nodes.remove(closest)

        return sol


    def triangle_swap(self):
        i1, i2, i3 = random.sample(range(0, len(self.solution)-1), 3)
        new_solution = self.solution[:]
        new_solution[i1], new_solution[i2], new_solution[i3] = new_solution[i2], new_solution[i3], new_solution[i1]
        return new_solution


    def swap(self):
        i1, i2 = random.sample(range(0, len(self.solution)-1), 2)
        new_solution = self.solution[:]
        new_solution[i1], new_solution[i2] = new_solution[i2], new_solution[i1]
        return new_solution


    def shuffle(self):
        new_solution = self.solution[:]
        i1, i2 = random.sample(range(0, len(new_solution)-1), 2)
        sub = new_solution[i1:i2]
        random.shuffle(sub)
        new_solution[i1:i2] = sub
        return new_solution    


    def get_distance(self,x1, y1, x2, y2) :
        return np.sqrt((x1-x2)**2 + abs(y1-y2)**2)


    def total_distance(self, solution = -1):
        if solution == -1:
            solution = self.solution
        distance = 0
        for i in range(len(solution)-1):
            fro = self.df.loc[solution[i]]
            to = self.df.loc[solution[i+1]]
            distance += self.get_distance(fro["x"], fro["y"], to["x"], to["y"])
        # make it a circle
        fro = self.df.loc[solution[len(solution)-1]]
        to = self.df.loc[solution[0]]
        distance += self.get_distance(fro["x"], fro["y"], to["x"], to["y"])
        return distance


    def plot_solution(self):
        for i in range(len(self.solution)-1):
            fro = self.df.loc[self.solution[i]]
            to = self.df.loc[self.solution[i+1]]
            plt.arrow(fro["x"], fro["y"], to["x"]-fro["x"], to["y"]-fro["y"])
        fro = self.df.loc[self.solution[len(self.solution)-1]]
        to = self.df.loc[self.solution[0]]
        plt.arrow(fro["x"], fro["y"], to["x"]-fro["x"], to["y"]-fro["y"])