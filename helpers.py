import numpy as np
import matplotlib.pyplot as plt
import random
import math


class AnnealTVS():
    def __init__(self, dataframe, repeats=1, K = 2, markov_length = 100, stopK = 0.1, alpha = 0.99,
        elementary = "2opt", verbose=False, alternate=False, secondary=-1, beta=0.05, gamma=1, 
        all_distances=False, return_solution=False):
        """A class that tries to solve a given TVS problem using Simulated Annealing.
        Args:
            dataframe (Pandas Dataframe): A dataframe that contains the coordinates of the TVS problem.
            repeats (int, optional): The amount of times the temperature is set to a high value. Defaults to 1.
            K (int, optional): Starting temperature. Defaults to 2.
            markov_length (int, optional): The length of the markov chain at every temperature. Defaults to 100.
            stopK (float, optional): The final temperature. Defaults to 0.1.
            alpha (float, optional): The amount by which the temperature is multiplied at every step. Should be lower than 1. Defaults to 0.99.
            elementary (str, optional): The elementary edit tried at every step. Defaults to "2opt".
            verbose (bool, optional): Whether or not to print a summary of every markov chain. Defaults to False.
            alternate (bool, optional): If true, the algorithm changes to a secondary elementary edit at the end of the run. Defaults to False.
            secondary (string, optional): The secondary the to change to. Defaults to -1.
            beta (float, optional): The amount at which to switch to the secondary strategy. Defaults to 0.05.
            gamma (int, optional): The amount the starting temperature gets multiplied with at every step. Defaults to 1.
            all_distances (Bool, optional): Choose whether you want to return all distances and temperatures.
            return_solution (Bool, optional): Choose whether to return the solution.
        """
        assert alpha < 1, "Alpha should be <1"

        self.df = dataframe
        self.K = K
        self.startK = K
        self.stopK = stopK
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.elementary = elementary
        self.start_strat = elementary
        self.verbose = verbose
        self.repeats = repeats
        self.markov_length = markov_length
        self.alternate = alternate
        self.all_distances = all_distances
        self.return_solution = return_solution

        if alternate:
            assert secondary != -1, "If you want to alternate, enter a secondary strategy"
            self.secondary = secondary
            
        # start with a simple solution
        self.dist_matrix = self.build_matrix()
        self.solution = self.nearest_neighbours()


    def run_sim(self):
        """Calls the simulated annealing algorithm.
        Returns:
            list: A list with the results of every run
        """
        results = []
        for i in range(self.repeats):
            results.append(self.sim_anneal())
            self.startK=self.startK * self.gamma
            self.K = self.startK
            self.elementary = self.start_strat
        return results


    def sim_anneal(self):
        """The Simulated annealing algorithm
        Raises:
            ValueError: The elementary edit should be one of the implemented strategies.
        Returns:
            Float: The final total distance of the path.
        """
        assert self.alpha < 1, "choose a smaller alpha"
        all_dist = []
        temperature = []
        old_dist = self.total_distance()
        i=0
        while(self.K>self.stopK):
            if self.verbose:
                lowered = 0
                raised = 0

            # run an epoch
            for i in range(self.markov_length):
                temperature.append(self.K)
                all_dist.append(self.total_distance())
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
                else:
                    # accept with probability depending on temperature
                    rand = np.random.random()
                    prob = math.exp((-1*(new_dist - old_dist)) / self.K )
                    if rand < prob:
                        if self.verbose:
                            raised += 1
                        self.solution = new_solution
                        old_dist = self.total_distance()
                    
            if self.verbose:
                print(f"temperature: {self.K}\ntimes lowered: {lowered}\ntimes raised:{raised}")
            # scale the cooling scheme to start lower every next simulation
            self.K *= self.alpha 
            # swap to secondary strategy if temperature is low enough
            if self.alternate:
                if self.K < (self.beta*self.startK):
                    self.elementary = self.secondary

        if self.all_distances:
            returns = [all_dist, temperature]
        else:
            returns = self.total_distance()
        if self.return_solution:
            returns = [returns, self.solution]
        return returns


    def nearest_neighbours(self):
        """Creates a starting point using a greedy nearest neighbour algorithm.
        Returns:
            List: A list with the order of the cities that are travelled.
        """
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
        """An triangle swap elementary edit.
        Returns:
            list: New solution.
        """
        i1, i2, i3 = random.sample(range(0, len(self.solution)-1), 3)
        new_solution = self.solution[:]
        new_solution[i1], new_solution[i2], new_solution[i3] = new_solution[i2], new_solution[i3], new_solution[i1]
        return new_solution


    def swap(self):
        """A simple swap elementary edit.
        Returns:
            list: New solution.
        """
        i1, i2 = random.sample(range(0, len(self.solution)-1), 2)
        new_solution = self.solution[:]
        new_solution[i1], new_solution[i2] = new_solution[i2], new_solution[i1]
        return new_solution


    def insert(self):
        """The insert of a element at a random location.
        Returns:
            list: New solution.
        """
        new_solution = self.solution[:]
        node = random.choice(new_solution)
        new_solution.remove(node)
        insert = random.randint(0, len(new_solution)-1)
        new_solution.insert(insert,node)
        return new_solution


    def shuffle(self):
        """Shuffles a part of the solution.
        Returns:
            list: New solution.
        """
        new_solution = self.solution[:]
        i1, i2 = random.sample(range(0, len(new_solution)-1), 2)
        sub = new_solution[i1:i2]
        random.shuffle(sub)
        new_solution[i1:i2] = sub
        return new_solution    


    def opt2(self):
        """Changes the order between two cities, effectively untangling any knots in between.
        Returns:
            list: New solution.
        """
        i = random.randint(0, len(self.solution)-2)
        k = i+random.randint(1, len(self.solution) -1 -i)
        new_solution = self.solution[0:i]

        for j in range((k-i), -1, -1):
            new_solution.append(self.solution[i+j])
        new_solution = new_solution + self.solution[(k+1):len(self.solution)]


        #assert len(new_solution) == len(self.solution)
        return new_solution


    def get_distance(self,x1, y1, x2, y2) :
        """Computes the Euclidean distance between two points.
        Args:
            x1 (int): X coordinate of the first city.
            y1 (int): Y coordinate of the first city.
            x2 (int): X coordinate of the second city.
            y2 (int): X coordinate of the second city.
        Returns:
            [float]: Distance.
        """
        return np.sqrt((x1-x2)**2 + abs(y1-y2)**2)


    def total_distance(self, solution = -1):
        """The total distance of a given solution
        Args:
            solution (int, optional): A solution. Not necessary as it will take the solution in the class. Defaults to -1.
        Returns:
            float: Total distance.
        """
        if solution == -1:
            solution = self.solution
        distance = 0
        for i in range(len(solution)-1):
            distance += self.dist_matrix[int(solution[i])-1][int(solution[i+1])-1]
        # make it a circle
        distance += self.dist_matrix[int(solution[i+1])-1][int(solution[0])-1]
        return distance


    def plot_solution(self):
        """Plots the path of the salesmen
        """
        self.df.plot.scatter("x", "y")
        for i in range(len(self.solution)-1):
            fro = self.df.loc[self.solution[i]]
            to = self.df.loc[self.solution[i+1]]
            plt.arrow(fro["x"], fro["y"], to["x"]-fro["x"], to["y"]-fro["y"])
        fro = self.df.loc[self.solution[len(self.solution)-1]]
        to = self.df.loc[self.solution[0]]
        plt.arrow(fro["x"], fro["y"], to["x"]-fro["x"], to["y"]-fro["y"])


    def build_matrix(self):
        """Builds a distance matrix of all the cities in the dataframe.
        Returns:
            [numpy matrix]: A matrix that contains the distance between two cities
        """
        matrix = np.zeros((len(self.df),len(self.df)))
        for row in range(len(self.df)):
            fro = self.df.loc[f"{row+1}"]
            for column in range(len(self.df)):
                if row != column:
                    to = self.df.loc[f"{column+1}"]
                    matrix[row][column] = self.get_distance(fro["x"], fro["y"], to["x"], to["y"])
        
        return matrix