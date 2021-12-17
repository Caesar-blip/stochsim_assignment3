import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import random
import math
from joblib import Parallel, delayed

class AnnealTVS():
    def __init__(self, dataframe, num_sim=1, K = 2, num_searches = 100, stopK = 0.1, alpha = 0.99,
        elementary = "triangle", verbose=False, alternate=False, secondary=-1, beta=0.05, gamma=1):
        
        self.df = dataframe
        self.K = K
        self.startK = K
        self.stopK = stopK
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.elementary = elementary
        self.verbose = verbose
        self.num_sim = num_sim
        self.num_searches = num_searches
        self.alternate = alternate

        if alternate:
            self.secondary = secondary
            
        # start with a simple solution
        self.dist_matrix = self.build_matrix()
        self.solution = self.nearest_neighbours()


    def run_sim(self):
        results = []
        for i in range(self.num_sim):
            results.append(self.sim_anneal())
            self.startK=self.startK * self.gamma
            self.K = self.startK
        return results


    def sim_anneal(self):
        assert self.alpha < 1, "choose a smaller alpha"
        all_dist = []
        old_dist = self.total_distance()
        i=0
        while(self.K>self.stopK):
            if self.verbose:
                lowered = 0
                raised = 0

            # run an epoch
            for i in range(self.num_searches):
                # elementary edit, triangle swap for now
                if self.elementary =="triangle":
                    new_solution = self.triangle_swap()
                elif self.elementary == "swap":
                    new_solution = self.swap()
                elif self.elementary == "shuffle":
                    new_solution = self.shuffle()
                elif self.elementary == "insert":
                    new_solution = self.insert()
                elif self.elementary == "2opt":
                    new_solution = self.opt2()
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
                    prob = math.exp((-1*abs(new_dist - old_dist)) / self.K )
                    if rand < prob:
                        if self.verbose:
                            raised += 1
                        self.solution = new_solution
                        old_dist = self.total_distance()
                        all_dist.append(self.total_distance())

            if self.verbose:
                print(f"temperature: {self.K}\ntimes lowered: {lowered}\ntimes raised:{raised}")
            # scale the cooling scheme to start lower every next simulation
            self.K *= self.alpha 
            # swap to secondary strategy if temperature is low enough
            if self.alternate:
                if self.K < (self.beta*self.startK):
                    self.elementary = self.secondary
            
        return self.solution


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


    def insert(self):
        new_solution = self.solution[:]
        node = random.choice(new_solution)
        new_solution.remove(node)
        insert = random.randint(0, len(new_solution)-1)
        new_solution.insert(insert,node)
        return new_solution


    def shuffle(self):
        new_solution = self.solution[:]
        i1, i2 = random.sample(range(0, len(new_solution)-1), 2)
        sub = new_solution[i1:i2]
        random.shuffle(sub)
        new_solution[i1:i2] = sub
        return new_solution    


    def opt2(self):
        i = random.randint(0, len(self.solution)-2)
        k = i+random.randint(1, len(self.solution) -1 -i)
        new_solution = self.solution[0:i]

        for j in range((k-i), -1, -1):
            new_solution.append(self.solution[i+j])
        new_solution = new_solution + self.solution[(k+1):len(self.solution)]


        #assert len(new_solution) == len(self.solution)
        return new_solution


    def get_distance(self,x1, y1, x2, y2) :
        return np.sqrt((x1-x2)**2 + abs(y1-y2)**2)


    def total_distance(self, solution = -1):
        if solution == -1:
            solution = self.solution
        distance = 0
        for i in range(len(solution)-1):
            distance += self.dist_matrix[int(solution[i])-1][int(solution[i+1])-1]
        # make it a circle
        distance += self.dist_matrix[int(solution[i+1])-1][int(solution[0])-1]
        return distance


    def plot_solution(self):
        self.df.plot.scatter("x", "y")
        for i in range(len(self.solution)-1):
            fro = self.df.loc[self.solution[i]]
            to = self.df.loc[self.solution[i+1]]
            plt.arrow(fro["x"], fro["y"], to["x"]-fro["x"], to["y"]-fro["y"])
        fro = self.df.loc[self.solution[len(self.solution)-1]]
        to = self.df.loc[self.solution[0]]
        plt.arrow(fro["x"], fro["y"], to["x"]-fro["x"], to["y"]-fro["y"])


    def build_matrix(self):
        matrix = np.zeros((len(self.df),len(self.df)))
        for row in range(len(self.df)):
            fro = self.df.loc[f"{row+1}"]
            for column in range(len(self.df)):
                if row != column:
                    to = self.df.loc[f"{column+1}"]
                    matrix[row][column] = self.get_distance(fro["x"], fro["y"], to["x"], to["y"])
        
        return matrix
