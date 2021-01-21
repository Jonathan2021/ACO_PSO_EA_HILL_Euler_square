from copy import deepcopy
import itertools
import random
from time import time
import numpy as np

def combinate(set1, set2):
    return list(itertools.product(set1, set2))

def matrixOfDict(side_length, dictionary):
        return [[dictionary.copy() for _ in range(side_length)] for _ in range(side_length)]

def makeDictIter(iterable, value = 0):
    dictionary = dict()
    for val in iterable:
        dictionary[val] = value
    return dictionary


def matrixDictApply(matrix, func):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            for key in matrix[i][j]:
                matrix[i][j][key] = func(matrix[i][j][key])
    return matrix

class Configuration: # Equivalent to an Individual
    def __init__(self, values, fitness): # probably shouldnt put fitness here but stored next to individual in EA etc.
        self._values = deepcopy(values)
        self._size = len(values)
        self._fit = fitness
        self.evaluate()

    @classmethod
    def copy(cls, config):
        return cls(config._values, config._fit)

    @classmethod
    def initial(cls, set1, set2, fitness_func):
        combos = combinate(set1, set2)
        length = len(set1)
        values = []
        for _ in range(length):
            row = []
            for _ in range(length):
                row.append(combos.pop(random.randint(0, len(combos) - 1)))
            values.append(row)
        return cls(values, fitness_func)
    
    @property
    def size(self):
        return self._size

    @property
    def values(self):
        return deepcopy(self._values)

    @property
    def fitness(self):
        return self._fitness

    def evaluate(self):
        self._fitness = self._fit(self)

    def swap(self, i, j, k, l):
        self._values[i][j], self._values[k][l] = \
            self._values[k][l], self._values[i][j]

    def __len__(self):
        return self.size

    def __eq__(self, other):
        if not isinstance(other, Configuration):
            return False
        if self._size != other.size:
            return False
        other_values = other.values
        for i in range(self._size):
            for j in range(self._size):
                if self._values[i][j] != other_values[i][j]:
                    return False
        return True

    def __str__(self):
        return '\n'.join('\t'.join('(%d, %d)' % (x[0], x[1]) for x in y) for y in self._values)


def swap(matrix, point_1, point_2):
    matrix[point_1[0]][point_1[1]], matrix[point_2[0]][point_2[1]] = matrix[point_2[0]][point_2[1]], matrix[point_1[0]][point_1[1]]


class Ant:
    def __init__(self, path, unused, side_size):
        self._side_size = side_size
        self._nb_cells = self._side_size * self._side_size
        self._path = path[:]
        self._current_config = None
        self._is_dead_end = False
        self._unused = unused
        self.evaluate() # Necessary ?

    @classmethod
    def initial(cls, set1, set2):
        combos = combinate(set1, set2)
        positionx, positiony = random.choices(range(len(set1)), k=2)
        value = combos.pop(random.randint(0, len(combos) - 1))
        path = [((positionx, positiony), value)]
        return cls(path, combos,  len(set1))

    @classmethod
    def copy(cls, ant):
        copy = cls(ant._path[:], ant._unused[:], ant._side_size)
        copy._is_dead_end = ant._is_dead_end
        copy._current_config = Configuration.copy(ant._current_config)
        return copy

    def isBetter(self, ant):
        return self.fitness < ant.fitness

    @property
    def current_config(self):
        if self._current_config is None:
            self._build_config()
        return self._current_config

    def _build_config(self):
        values = [[None for _ in range(self._side_size)] for _ in range(self._side_size)]
        indexes, path_values = zip(*self._path)
        for i in range(len(indexes)):
            x, y = indexes[i]
            values[x][y] = path_values[i]
        self._current_config = self._dummyConfig(values)

    @staticmethod
    def _dummyConfig(values):
        return Configuration(values, lambda x : 1)

    @staticmethod
    def _isOkInsert(values, value, positionx, positiony):
        row = [x for x in values[positionx] if x]
        col = [values[i][positiony] for i in range(len(values)) if values[i][positiony]]
        rowfirst, rowsecond = zip(*row) if row else ([], [])
        colfirst, colsecond = zip(*col) if col else ([], [])
        first = list(rowfirst) + list(colfirst)
        second = list(rowsecond) + list(colsecond)
        return (value[0] not in first and value[1] not in second)

    def nextMoves(self):
        if self._is_dead_end:
            return []
        values = self.current_config.values
        new = []
        path_indexes,_ = zip(*self._path)
        for i in range(self._side_size):
            for j in range(self._side_size):
                if (i,j) not in path_indexes:
                    for unused in self._unused:
                        if self._isOkInsert(values, unused, i, j):
                            new = [((i,j), unused)]
        if not new:
            self._is_dead_end = True
        return new

    def _insertMove(self, move):
        (x, y), value = move
        self._unused.remove(value)
        self._path.append(move)
        values = self._current_config.values
        values[x][y] = value
        self._current_config = self._dummyConfig(values)
        self.evaluate()


    def distMove(self, move):
        dummy = self.copy(self)
        dummy._insertMove(move)
        return len(dummy.nextMoves())


    @staticmethod
    def _getMoveFromRepartitionFunc(matrix):
        curSum = 0 # Sort of repartition function
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                for key in matrix[i][j]:
                    curSum += matrix[i][j][key]
                    if random.random() < curSum:
                        return ((i,j), key)
            
    def _makeCleanProbaMatrix(self):
        unused_dic = makeDictIter(self._unused, 0)
        return matrixOfDict(self._side_size, unused_dic)

    def addMove(self, q0, trace, alpha, beta):
        nextSteps = self.nextMoves()
        if self._is_dead_end:
            return False
        p = self._makeCleanProbaMatrix()
        unused_dic = p[0][0]
        for move in nextSteps:
            (x, y), value = move
            p[x][y][value] = self.distMove(move)

        # Originally traverse matrix again in another function but it gets costy
        # Could make it in a separate function but couldn't bother
        maxProba = float('-inf')
        maxValue = None
        rowMax = -1
        colMax = -1
        sumProba = 0

        (x,y), prev_value = self._path[-1]
        for i in range(self._side_size):
            for j in range(self._side_size):
                for val in unused_dic:
                    current = (p[i][j][val]**beta)*(trace[x][y][prev_value][i][j][val]**alpha)
                    p[i][j][val] = current
                    if maxProba < current:
                        maxProba = current
                        maxValue = val
                        rowMax = i
                        colMax = j
                    sumProba += current

        if random.random() < q0:
            self._insertMove(((rowMax, colMax), maxValue))
        else:
            if not sumProba:
                self._insertMove(random.choice(nextSteps))
            else:
                matrixDictApply(p, lambda x : x/sumProba)
                self._insertMove(self._getMoveFromRepartitionFunc(p))
        return True
                            
            

    def evaluate(self):
        self._fitness = len(self._unused) + 1

    @property
    def fitness(self):
        return self._fitness

    @property
    def path(self):
        return self._path[:]

    def __str__(self):
        return '\n'.join('\t'.join('(%d, %d)' % (x[0], x[1]) if x else "(?, ?)" for x in y) for y in self._current_config.values)
        

class Particle(Configuration):
    def __init__(self, values, fitness):
        super().__init__(values, fitness)
        self._bestValues = self.values
        self._bestFitness = self.fitness
        self.velocity = [[(0., 0.) for _ in range(self.size)] for _ in range(self.size)]
    
    @property
    def bestValues(self):
        return self._bestValues

    @property
    def bestFitness(self):
        return self._bestFitness

    @Configuration.values.setter
    def values(self, newValues):
        self._values = deepcopy(newValues)
        self.evaluate()
        #FIXME < or > is problem specific. But if I pass a problem as an attribute then it becomes kinda circular ?
        if (self.fitness < self.bestFitness):
            self._bestValues = self.values
            self._bestFitness = self.fitness

def printMatrix(matrix, func): 
    print('\n'.join('\t'.join(func(x) for x in y) for y in matrix))
    
  
def printMatrixTuple(matrix):
    printMatrix(matrix, lambda x : ('%f, %f') % (x[0], x[1]))

def checkAllUnique(array):
    seen = set()
    return not any(i in seen or seen.add(i) for i in array)

class Hill:
    def __init__(self, fitness):
        self._fitness = fitness

    def _nextConfig(self, config, i, j):
        nextC = []
        for k in range(len(config)):
            for l in range(len(config)):
                maybe_next = config.values
                swap(maybe_next, (i, j), (k, l))
                nextC.append(Configuration(maybe_next, self._fitness))
        return nextC

    def expand(self, currentConfig):
        myList = []
        for j in range(len(currentConfig)):
            for k in range(len(currentConfig)):
                myList.extend(self._nextConfig(currentConfig, j, k))
        return myList

class ACO:
    def __init__(self, alpha, beta, q0, rho):
        self._alpha = alpha
        self._beta = beta
        self._q0 = q0
        self._rho = rho

    original_trace_weight = 1

    @staticmethod
    def _population(set1, set2, numberAnts):
        return [Ant.initial(set1, set2) for _ in range(numberAnts)]
    
    @staticmethod
    def makeTrace(set1, set2):
        side_length = len(set1)
        dictionary = makeDictIter(combinate(set1, set2), 1)
        return matrixDictApply(matrixOfDict(side_length, dictionary), lambda x: matrixOfDict(side_length, dictionary))

    @staticmethod
    def _traceApply(trace, func):
        matrixDictApply(trace, lambda x: matrixDictApply(x, func))

    def _updateTrace(self, antSet, trace):
        numberAnts = len(antSet)
        dTrace = [ 1.0 / antSet[i].fitness for i in range(numberAnts)]

        self._traceApply(trace, lambda x: (1 - self._rho) * x)

        for i in range(numberAnts):
            curpath = antSet[i].path
            for j in range(len(curpath) - 1):
                (x1, y1), val1 = curpath[j]
                (x2, y2), val2 = curpath[j + 1]
                self._traceApply(trace, lambda x: x + dTrace[i])

    def _moveAnts(self, antSet, max_path_length, trace):
        for _ in range(max_path_length):
            for ant in antSet:
                ant.addMove(self._q0, trace, self._alpha, self._beta)

    @staticmethod
    def _getBestAnt(antSet):
        return max(antSet, key = lambda x: x.fitness)

    def iteration(self, set1, set2, numberAnts, trace):
        side_size = len(set1)
        antSet = self._population(set1, set2, numberAnts)

        # Move ants
        self._moveAnts(antSet, side_size * side_size, trace)
        
        # Update trace
        self._updateTrace(antSet, trace)

        # return best ant path
        return self._getBestAnt(antSet)


class PSO:
    def __init__(self, inertia, cognitive, social, fitness):
        self._inertia = inertia
        self._cognitive = cognitive
        self._social = social
        self._fitness = fitness

    def population(self, count, set1, set2): # Since we will have duplicates in any case, shouldn't care if there are some at the start
        return [Particle.initial(set1, set2, self._fitness) for _ in range(count)]

    def selectNeighbors(self, pop, nSize):
        pop_length = len(pop)
        if nSize > pop_length:
            nSize = pop_length
        neighbors = []
        for i in range(len(pop)):
            neighbors.append(random.sample(range(pop_length), k = nSize)) # can have itself as a neighbor
        return neighbors
        
    def _updateVelo(self, row, col, tochange, ref):
        velo_tuple = ()
        for i in range(2):
            newVelocity = self._inertia * tochange.velocity[row][col][i]
            newVelocity = newVelocity + self._social * random.random() * \
                (ref.values[row][col][i] - tochange.values[row][col][i])
            newVelocity = newVelocity + self._cognitive * random.random() * \
                (tochange.bestValues[row][col][i] - tochange.values[row][col][i])
            velo_tuple += (newVelocity,)
        tochange.velocity[row][col] = velo_tuple

    def _updateValues(self, tochange):
        newValues = []
        for j in range(len(tochange)):
            row = []
            for k in range(len(tochange)):
                val_tuple = ()
                for l in range(2):
                    newVal = round(tochange.values[j][k][l] + tochange.velocity[j][k][l])
                    val_tuple += (newVal,)
                row.append(val_tuple)
            newValues.append(row)
        tochange.values = newValues

    #FIXME Kind of a bad technique. Correct solutions are not 'next to each other'
    def iteration(self, pop, neighbors):
        bestNeighbors = []
        for i in range(len(pop)):
            bestNeighbors.append(neighbors[i][0])
            for j in range(1, len(neighbors[i])):
                if (pop[bestNeighbors[i]].fitness > pop[neighbors[i][j]].fitness):
                    bestNeighbors[i] = neighbors[i][j]
        for i in range(len(pop)):
            cur_config = pop[i]
            cur_best = pop[bestNeighbors[i]]
            # since len of particle is len of values = len of velocity
            for j in range(len(cur_config)):
                for k in range(len(cur_config)):
                    self._updateVelo(j, k, cur_config, cur_best)
        for i in range(len(pop)):
            self._updateValues(pop[i])
        return pop

class EA:
    def __init__(self, fitness):
        self._fitness = fitness

    def population(self, count, set1, set2): #FIXME Duplicate of PSO population. Should make a general one
        return [Configuration.initial(set1, set2, self._fitness) for _ in range(count)]

    def mutate(self, individual, pM):
        if pM > random.random():
            i = random.randint(0, len(individual) - 1)
            j = random.randint(0, len(individual) - 1)
            while True:
                k = random.randint(0, len(individual) - 1)
                l = random.randint(0, len(individual) - 1)
                if k != i or l != j:
                    break
            individual.swap(i, j, k, l)
        return individual

    def crossover(self, parent1, parent2):
        values = []
        scores = []
        random_ceiling = 1
        for row in parent1.values:
            for el in row:
                random_ceiling = random.uniform(0, random_ceiling)
                values.append(el)
                scores.append(random_ceiling)
        random_ceiling = 1
        for row in parent2.values:
            for el in row:
                random_ceiling = random.uniform(0, random_ceiling)
                scores[values.index(el)] += random_ceiling
        zipped = list(zip(values, scores))
        zipped.sort(key=lambda x: x[1])
        values, _ = zip(*zipped)
        size = len(parent1)
        values = [[values[i * size + j] for j in range(size)] for i in range(size)]
        return Configuration(values, self._fitness)

    def iteration(self, pop, pM, k):
        indexes = random.sample(range(len(pop)), k=k) # Tournament selection
        zipped = [(pop[index], index) for index in indexes]
        zipped.sort(key=lambda x : x[0].fitness)
        while True: # Ranking selection
            random_indexes = random.choices(range(k), k=k)
            random_indexes = list(set(random_indexes))
            random_indexes.sort()
            if len(random_indexes) >= 2:
                break;
        parent1 = zipped[random_indexes[0]]
        parent2 = zipped[random_indexes[1]]
        c = self.crossover(parent1[0], parent2[0])
        c = self.mutate(c, pM)
        f1 = parent1[0].fitness
        f2 = parent2[0].fitness
        fc = c.fitness
        if (f1 > f2) and (f1 > fc):
            pop[parent1[1]] = c
        elif (f2 > f1) and (f2 > fc):
            pop[parent2[1]] = c
        return pop

class Problem:
    def __init__(self, set1, set2):
        self.__set1 = set1[:]
        self.__set2 = set2[:]

    @staticmethod
    def isFinal(config): # Not needed
        visited = []
        values = config.values
        for row in values:
            firsts = [el[0] for el in row]
            seconds = [el[1] for el in row]
            if not checkAllUnique(firsts) or not checkAllUnique(seconds):
                return False
        transposed = np.transpose(values)
        for row in transposed:
            firsts = [el[0] for el in row]
            seconds = [el[1] for el in row]
            if not checkAllUnique(firsts) or not checkAllUnique(seconds):
                return False
        for row in values:
            if any(el in visited for el in row):
                return False
            visited.extend(row)
        return True

    def getSets(self):
        return (self.__set1[:], self.__set2[:])

    def setSets(self, set1, set2):
        self.__set1 = set1[:]
        self.__set2 = set2[:]

    def fitness(self, config):
        res = 0
        values = config.values
        transposed = [[values[j][i] for j in range(len(values))] for i in range(len(values[0]))] 
        matrixes = [values, transposed]
        for mat in matrixes:
            for row in mat:
                firsts = [el[0] for el in row]
                seconds = [el[1] for el in row]
                res += len(firsts) - len(set(firsts))
                res += len(seconds) - len(set(seconds))
                res += sum(1 for x in firsts if x not in self.__set1) # These lines are relevant ony for PSO. Could maybe do this only in PSO
                res += sum(1 for x in seconds if x not in self.__set2)
        return res

    @staticmethod
    def getBestConfig(list_config):
        return min(list_config, key=lambda x: x.fitness)

    @staticmethod
    def isBetterFitness(a, b):
        return a < b

    @staticmethod
    def isOptimalSolution(config):
        return not config.fitness

class Controller:
    def __init__(self, problem, eaconf, psoconf, acoconf):
        self._problem = problem
        self._eaconf = eaconf
        self._psoconf = psoconf
        self._acoconf = acoconf

    def Hill(self, verbose=True):
        hill = Hill(self._problem.fitness)
        set1, set2 = self._problem.getSets()
        best_config = Configuration.initial(set1, set2, self._problem.fitness)
        count = 0
        while True:
            if verbose:
                print(f"******* iteration {count} *********")
            neighbours = hill.expand(best_config)
            potential = Problem.getBestConfig(neighbours) 
            if Problem.isBetterFitness(potential.fitness, best_config.fitness):
                if verbose:
                    print(f"Climbing from fitness {best_config.fitness} to {potential.fitness}")
                best_config = potential
            else:
                break
            count +=1
        return (best_config, count)

    def EA(self, verbose=True):
        config = self._eaconf
        # population size
        dimPop = config.dimPop
        # mutation proba
        pM = config.pM
        # iterations
        iterations = config.iterations
        # Pressure (size of tournament sample
        pressure = config.pressure

        ea = EA(self._problem.fitness)
        set1, set2 = self._problem.getSets()
        pop = ea.population(dimPop, set1, set2) #maybe shouldnt be part of ea class
        curbest = pop[0]
        curbestFitness = curbest.fitness
        for i in range(iterations):
            pop = ea.iteration(pop, pM, pressure)
            best = self._problem.getBestConfig(pop)
            if self._problem.isBetterFitness(best.fitness, curbestFitness):
                if verbose:
                    print(f"******* iteration {i+1} **************")
                    print(f"New best config at iteration {i}")
                    print(best)
                    print(f"Fitness : {best.fitness}")
                curbestFitness = best.fitness
                curbest = best
        return (curbest, pop)

    def PSO(self, verbose=True):
        config = self._psoconf
        #number of particles
        noParticles = config.dimPop
        #size of neighborhood
        dimNeighbors = config.dimNeighbors
        # sets
        set1, set2 = self._problem.getSets()
        # iterations
        iterations = config.iterations

        pso = PSO(inertia = config.inertia, cognitive = config.cognitive, social = config.social, fitness = self._problem.fitness)
        pop = pso.population(noParticles, set1, set2)
        neighbors = pso.selectNeighbors(pop, dimNeighbors)

        curbest = pop[0]
        curbestFitness = curbest.fitness
        for i in range(iterations):
            if verbose:
                print(f"******* iteration {i+1} **************")
            pop = pso.iteration(pop, neighbors)
            best = self._problem.getBestConfig(pop)
            if self._problem.isBetterFitness(best.fitness, curbestFitness):
                if verbose:
                    print(f"New 'best' config at iteration {i}")
                    print(best)
                    print(f"Fitness : {best.fitness}")
                curbestFitness = best.fitness
                curbest = best
        return (curbest, pop)
    
    def ACO(self, verbose=True):
        config = self._acoconf
        set1, set2 = self._problem.getSets()
        iterations = config.iterations
        numberAnts = config.dimPop
        bestAnt = Ant.initial(set1, set2)
        bestAntiter = 0
        aco = ACO(config.alpha, config.beta, config.q0, config.rho)
        trace = ACO.makeTrace(set1, set2)
        if verbose:
            print("ACO started, please be patient:")
        for i in range(iterations):
            print(f"*********** Iteration {i + 1} ***************")
            sol = aco.iteration(set1, set2, numberAnts, trace)
            if sol.isBetter(bestAnt):
                bestAnt = sol
                bestAntiter = i + 1
                if verbose:
                    print("New best ant")
                    print(bestAnt)
                    print("Fitness ", bestAnt.fitness - 1)
        return (bestAnt, bestAntiter)


class GenerationConfig:
    def __init__(self, dimPop, iterations):
        self._dimPop = dimPop
        self._iterations = iterations

    @property
    def dimPop(self):
        return self._dimPop

    @property
    def iterations(self):
        return self._iterations

    @dimPop.setter
    def dimPop(self, dimPop):
        self._dimPop = dimPop

    @iterations.setter
    def iterations(self, iterations):
        iterations = max(iterations, 0)
        self._iterations = iterations

    def __str__(self):
        s = f"Number of iterations: {self._iterations}\n"
        s += f"Population size: {self._dimPop}\n"
        return s

class PSOConfig (GenerationConfig):
    def __init__(self, dimPop, dimNeighbors,inertia, cognitive, social, iterations):
        super().__init__(dimPop, iterations)
        self._dimNeighbors = dimNeighbors
        self._cognitive = cognitive
        self._inertia = inertia
        self._social = social

    @property
    def dimNeighbors(self):
        return self._dimNeighbors

    @property
    def inertia(self):
        return self._inertia

    @property
    def cognitive(self):
        return self._cognitive

    @property
    def social(self):
        return self._social

    @GenerationConfig.dimPop.setter
    def dimPop(self, dimPop):
        dimPop = max(dimPop, 2)
        self._dimNeighbors = min(self._dimNeighbors, dimPop)
        GenerationConfig.dimPop.fset(self, dimPop)

    @dimNeighbors.setter
    def dimNeighbors(self, dimNeighbors): #FIXME Check if > dimPop and reduce here ?
        dimNeighbors = max(dimNeighbors, 2)
        dimNeighbors = min(dimNeighbors, self._dimPop)
        self._dimNeighbors = dimNeighbors

    @inertia.setter
    def inertia(self, inertia):
        self._inertia = inertia
    
    @cognitive.setter
    def cognitive(self, cognitive):
        self._cognitive = cognitive

    @social.setter
    def social(self, social):
        self._social = social

    def __str__(self):
        s = super().__str__()
        s += f"Number of neighbors: {self._dimNeighbors}\n"
        s += f"Inertia factor: {self._inertia}\n" 
        s += f"Cognition factor: {self._cognitive}\n"
        s += f"Social factor: {self._social}\n"
        return s

class ACOConfig(GenerationConfig):
    def __init__(self, dimPop, alpha, beta, rho, q0, iterations):
        super().__init__(dimPop, iterations)
        self._alpha = alpha
        self._beta = beta
        self._rho = rho
        self._q0 = q0

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def rho(self):
        return self._rho

    @property
    def q0(self):
        return self._q0

    @alpha.setter
    def alpha(self, val):
        val = max(val, 0.0)
        self._alpha = val

    @beta.setter
    def beta(self, val):
        val = max(val, 0.0)
        self._beta = val

    @rho.setter
    def rho(self, val):
        val = max(val, 0.0)
        self._rho = val

    @q0.setter
    def q0(self, val):
        val = max(val, 0.0)
        self._q0 = val

    def __str__(self):
        s = super().__str__()
        s += f"alpha: {self._alpha}\n"
        s += f"beta: {self._beta}\n" 
        s += f"rho: {self._rho}\n"
        s += f"q0: {self._q0}\n"
        return s


class EAConfig (GenerationConfig):
    def __init__(self, dimPop, pM, pressure, iterations):
        super().__init__(dimPop, iterations)
        self._pM = pM
        self._pressure = pressure

    @property
    def pM(self):
        return self._pM

    @property
    def pressure(self):
        return self._pressure

    @GenerationConfig.dimPop.setter
    def dimPop(self, dimPop):
        dimPop = max(dimPop, 2)
        self._pressure = min(self._pressure, dimPop)
        GenerationConfig.dimPop.fset(self, dimPop)
    
    @pM.setter
    def pM(self, pM):
        pM = max(pM, 0)
        pM = min(pM, 1)
        self._pM = pM

    @pressure.setter
    def pressure(self, pressure):
        pressure = max(pressure, 2)
        pressure = min(pressure, self._dimPop)
        self._pressure = pressure

    def __str__(self):
        s = super().__str__()
        s += f"Mutation probability: {self._pM}\n"
        s += f"Pressure: {self._pressure}\n"
        return s

def get_input(msg):
    return input((msg if msg is not None else "") + ">> ")

def getInt(msg):
    return int(get_input(msg))

def getFloat(msg):
    return float(get_input(msg))

def safe(fn, msg, default):
    try:
        return fn(msg)
    except KeyboardInterrupt:
        raise
    except:
        print(f"Invalid entry, the implicit value is still {default}")
        return default


class Menu:
    def __init__(self, title, description, choices):
        self._title = title
        self._description = description
        self._choices = choices

    def _printDescription(self):
        print(self._description)

    def printMainMenu(self):
        print("***************** ", self._title, " **********************")
        self._printDescription()
        print("0 - exit") # FIXME this is hard coded
        l_nb = 1
        for legend,_ in self._choices:
            print(l_nb, end = " - ")
            print(legend)
            l_nb += 1
    
    def run(self):
        runM=True
        while runM:
            self.printMainMenu()
            try:
                command = getInt("Enter option")
            except Exception as e:
                print("Invalid command")
                continue
            if command == 0:
                runM = False
                continue
            try:
                self._choices[command - 1][1](self)
            except IndexError:
                print("Incorrect option number")
                continue

class GenerationConfBuilder(Menu):
    def __init__(self, conf):
        super().__init__(self.menu_title, conf.__str__(), self.choices)
        self._conf = conf

    def readPopSubMenu(self):
        default = self._conf.dimPop
        n = safe(getInt, f"Input the size of population (implicit={default}) > 1", default)
        self._conf.dimPop = n

    def readIterSubMenu(self):
        default = self._conf.iterations
        n = safe(getInt, f"Input the number of iterations (implicit={default}) > 0", default)
        self._conf.iterations = n

    def _printDescription(self):
        print(self._conf)

    choices = [("read the number of iterations", readIterSubMenu),
                ("read the population size", readPopSubMenu)]

    menu_title = "Generation Menu"

class ACOConfBuilder(GenerationConfBuilder):
    def __init__(self, conf):
        super().__init__(conf)

    def readAlphaSubMenu(self):
        default = self._conf.alpha
        n = safe(getFloat, f"Input alpha (implicit={default}) > 0", default)
        self._conf.alpha = n

    def readBetaSubMenu(self):
        default = self._conf.beta
        n = safe(getFloat, f"Input beta (implicit={default}) > 0", default)
        self._conf.beta = n

    def readRhoSubMenu(self):
        default = self._conf.rho
        n = safe(getFloat, f"Input rho (implicit={default}) > 0", default)
        self._conf.rho = n

    def readQ0SubMenu(self):
        default = self._conf.q0
        n = safe(getFloat, f"Input q0 (implicit={default}) > 0", default)
        self._conf.q0 = n

    menu_title = "ACO Menu"

    choices = GenerationConfBuilder.choices + \
            [("read alpha", readAlphaSubMenu),
                    ("read beta", readBetaSubMenu),
                    ("read rho", readRhoSubMenu),
                    ("read q0", readQ0SubMenu)]

class PSOConfBuilder(GenerationConfBuilder):
    def __init__(self, conf):
        super().__init__(conf)

    def readNeighborsSubMenu(self):
        default = self._conf.dimNeighbors
        n = safe(getInt, f"Input the size of neighbors (implicit={default} in [2, {self._conf.dimPop}]", default)
        self._conf.dimNeighbors = n

    def readIntertiaSubMenu(self):
        default = self._conf.inertia
        n = safe(getFloat, f"Input the inertia factor (implicit={default})", default)
        self._conf.inertia = n

    def readCognitiveSubMenu(self):
        default = self._conf.cognitive
        n = safe(getFloat, f"Input the cognition factor (implicit={default})", default)
        self._conf.cognitive = n

    def readSocialSubMenu(self):
        default = self._conf.social
        n = safe(getFloat, f"Input the social factor (implicit={default})", default)
        self._conf.social = n
    
    menu_title = "PSO Menu"

    choices = GenerationConfBuilder.choices + \
            [("read the number of neighbors", readNeighborsSubMenu),
                    ("read the interta factor", readIntertiaSubMenu),
                    ("read the cognitive factor", readCognitiveSubMenu),
                    ("read the social factor", readSocialSubMenu)]

class EAConfBuilder(GenerationConfBuilder):
    def __init__(self, conf):
        super().__init__(conf)

    def readPressureSubMenu(self):
        default = self._conf.pressure
        n = safe(getInt, f"Input the selection pressure (implicit={default}) lowest : 2 highest : {self._conf.dimPop}", default)
        self._conf.pressure = n

    def readMutSubMenu(self):
        default = self._conf.pM
        n = safe(getFloat, f"Input the probability of a mutation happening (implicit={default}) in [0, 1]", default)
        self._conf.pM = n

    menu_title = "EA Menu"

    choices = GenerationConfBuilder.choices + \
            [("read the mutation probability", readMutSubMenu),
                    ("read the selection recombination pressure", readPressureSubMenu)]

def EAValidationTests(nbEval=1000, nbRuns=30, dimPop=40, pM=0.02, pressure = 5, setSize=3):
    s = "***************** Validation Tests EA **********************\n"
    eaconf = EAConfig(dimPop = dimPop, pM = pM, pressure = pressure, iterations = nbEval)
    print(eaconf)
    print(f"Size of square {setSize} x {setSize}")
    sets = range(1, setSize + 1)
    problem = Problem(sets, sets)
    controller = Controller(problem, eaconf, None, None)
    fitnesses = []
    for i in range(nbRuns):
        print(f"Run {i} / {nbRuns}")
        run_fit = controller.EA(verbose=False)[0].fitness
        print("Best fitness ", run_fit)
        fitnesses.append(run_fit)
    std = np.std(fitnesses)
    mean = np.mean(fitnesses)
    print("Mean of fitness : ", mean)
    print("Standard deviation of fitness : ", std)

def PSOValidationTests(nbEval=100, nbRuns=30, dimPop=40, dimNeighbors=20, inertia = 0.5, cognitive = 1.0, social = 1.0, setSize=3):
    print("***************** Validation Tests PSO **********************\n")
    psoconf = PSOConfig(dimPop = dimPop, dimNeighbors = dimNeighbors, inertia = inertia, cognitive = cognitive, social = social, iterations = nbEval)
    print(psoconf)
    print(f"Size of square {setSize} x {setSize}")
    sets = range(1, setSize + 1)
    problem = Problem(sets, sets)
    controller = Controller(problem, None, psoconf, None)
    fitnesses = []
    for i in range(nbRuns):
        print(f"Run {i + 1} / {nbRuns}")
        run_fit = controller.PSO(verbose=False)[0].fitness
        print("Best fitness: ", run_fit)
        fitnesses.append(run_fit)
    std = np.std(fitnesses)
    mean = np.mean(fitnesses)
    print("Mean of fitness : ", mean)
    print("Standard deviation of fitness : ", std)

class UI(Menu):
    def __init__(self):
        super().__init__("Main menu", "", self.choices)
        self._initSize = 3
        self._eaconf = EAConfig(dimPop = 100, pM = 0.02, pressure = 5, iterations = 10000)
        self._psoconf = PSOConfig(dimPop = 100, dimNeighbors = 20, inertia = 0.5, cognitive = 1., social = 1., iterations = 100)
        self._acoconf = ACOConfig(dimPop = 10, alpha = 1.8, beta = 0.9, rho = 0.05, q0 = 0.5, iterations = 100)

        default_set = range(1, self._initSize + 1)
        self._problem = Problem(default_set, default_set)
        self._contr = Controller(self._problem, self._eaconf, self._psoconf, self._acoconf)

    menu_title = "Main Menu"
    
    def _printDescription(self):
        print(f"Euler Square of size {self._initSize} x {self._initSize}")

    def readSizeSubMenu(self):
        default = self._initSize
        n = safe(getInt, f"Input the side length of the matrix (implicit={default})", default)
        self._initSize = n
        new_set = range(1, self._initSize + 1)
        self._problem.setSets(new_set, new_set)

    def buildEA(self):
        EAConfBuilder(self._eaconf).run()

    def buildPSO(self):
        PSOConfBuilder(self._psoconf).run()

    def buildACO(self):
        ACOConfBuilder(self._acoconf).run()

    def findSol(self, func):
        startClock = time()
        res = func()
        endClock = time() - startClock
        print('execution time = ', endClock, " seconds")
        return res

    @staticmethod
    def printSolution(solution):
        print("Solution found:")
        print(solution)
        print(f"fitness of solution : {solution.fitness}")

    def getOptimals(self, solutions):
        opts = []
        for sol in solutions:
            if self._problem.isOptimalSolution(sol) and sol not in opts:
                opts.append(sol)
        return opts

    @staticmethod
    def printSolutions(solutions):
        for i in range(len(solutions)):
            print(f"******* Solution n° {i + 1} ************")
            print(solutions[i])

    def findSolHill(self):
        sol, it = self.findSol(self._contr.Hill)
        UI.printSolution(sol)
        print(f"Number of iterations : {it}")

    def findPopBased(self, func):
        config, pop = self.findSol(func)
        UI.printSolution(config)
        opt = self.getOptimals(pop)
        nbOpt = len(opt)
        print(f"{nbOpt} correct solution to problem are in the population")
        if nbOpt and not safe(getInt, "type 0 to show all solutions found (implicit=0)", 0):
            UI.printSolutions(opt)
        
    def findSolPSO(self):
        self.findPopBased(self._contr.PSO)

    def findSolEA(self):
        self.findPopBased(self._contr.EA)

    def findSolACO(self):
        ant, iterations = self.findSol(self._contr.ACO)
        print(ant)
        print(f"Fitness : {ant.fitness - 1} found in {iterations} iterations")

    def ValidationEA(self):
        EAValidationTests(setSize=self._initSize)
        
    def ValidationPSO(self):
        PSOValidationTests(setSize=self._initSize)

    choices = [("read the size of a side of the square", readSizeSubMenu),
        ("EA option configuration", buildEA),
        ("PSO option configuration", buildPSO),
        ("ACO option configuration", buildACO),
        ("Find a solution with Hill", findSolHill),
        ("Find a solution with EA", findSolEA),
        ("Find a solution with PSO", findSolPSO),
        ("Find a solution with ACO", findSolACO),
        ("Validation tests EA", ValidationEA),
        ("Validation tests PSO", ValidationPSO)]



def main():
    ui = UI()
    ui.run()

main()
