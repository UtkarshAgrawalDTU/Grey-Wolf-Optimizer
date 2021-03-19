import numpy as np
import math

def prod(it):
    p = 1
    for n in it:
        p *= n
    return p


def Ufun(x, a, k, m):
    y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
    return y


class F1:
    def __init__(self, num_dimensions):
        self.dimensions = num_dimensions
        self.low = -100
        self.high = 100
        self.global_optimum_solution = 0 

    def get_func_val(self, x):
        s = np.sum(x ** 2)
        return s

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions
        




class F2:
    def __init__(self, num_dimensions):
        self.dimensions = num_dimensions
        self.low = -10
        self.high = 10
        self.global_optimum_solution = 0 

    def get_func_val(self, x):
        o = sum(abs(x)) + prod(abs(x))
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions


 
 
class F3:
    def __init__(self, num_dimensions):
        self.dimensions = num_dimensions
        self.low = -100
        self.high = 100
        self.global_optimum_solution = 0 

    def get_func_val(self, x):
        dim = len(x) + 1
        o = 0
        for i in range(1, dim):
            o = o + (np.sum(x[0:i])) ** 2
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions

    


class F4:
    def __init__(self, num_dimensions):
        self.dimensions = num_dimensions
        self.low = -100
        self.high = 100
        self.global_optimum_solution = 0 

    def get_func_val(self, x):
        o = max(abs(x))
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions


class F5:
    def __init__(self, num_dimensions):
        self.dimensions = num_dimensions
        self.low = -30
        self.high = 30
        self.global_optimum_solution = 0 

    def get_func_val(self, x):
        dim = len(x)
        o = np.sum(
            100 * (x[1:dim] - (x[0 : dim - 1] ** 2)) ** 2 + (x[0 : dim - 1] - 1) ** 2
        )
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions




class F6:
    def __init__(self, num_dimensions):
        self.dimensions = num_dimensions
        self.low = -100
        self.high = 100
        self.global_optimum_solution = 0 

    def get_func_val(self, x):
        o = np.sum(abs((x + 0.5)) ** 2)
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions
  



class F7:
    def __init__(self, num_dimensions):
        self.dimensions = num_dimensions
        self.low = -1.28
        self.high = 1.28
        self.global_optimum_solution = 0 

    def get_func_val(self, x):
        dim = len(x)

        w = [i for i in range(len(x))]
        for i in range(0, dim):
            w[i] = i + 1
        o = np.sum(w * (x ** 4)) + np.random.uniform(0, 1)
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions



class F8:
    def __init__(self, num_dimensions):
        self.dimensions = num_dimensions
        self.low = -500
        self.high = 500
        self.global_optimum_solution = -12569.5

    def get_func_val(self, x):
        o = sum(-x * (np.sin(np.sqrt(abs(x)))))
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions
   

class F9:
    def __init__(self, num_dimensions):
        self.dimensions = num_dimensions
        self.low = -5.12
        self.high = 5.12
        self.global_optimum_solution = 0

    def get_func_val(self, x):
        dim = len(x)
        o = np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x)) + 10 * dim
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions



class F10:
    def __init__(self, num_dimensions):
        self.dimensions = num_dimensions
        self.low = -32
        self.high = 32
        self.global_optimum_solution = 0

    def get_func_val(self, x):
        dim = len(x)
        o = (
            -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim))
            - np.exp(np.sum(np.cos(2 * math.pi * x)) / dim)
            + 20
            + np.exp(1)
        )
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions




class F11:
    def __init__(self, num_dimensions):
        self.dimensions = num_dimensions
        self.low = -600
        self.high = 600
        self.global_optimum_solution = 0

    def get_func_val(self, x):
        w = [i for i in range(len(x))]
        w = [i + 1 for i in w]
        o = np.sum(x ** 2) / 4000 - prod(np.cos(x / np.sqrt(w))) + 1
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions
    


class F12:
    def __init__(self, num_dimensions):
        self.dimensions = num_dimensions
        self.low = -50
        self.high = 50
        self.global_optimum_solution = 0

    def get_func_val(self, x):
        dim = len(x)
        o = (math.pi / dim) * (
            10 * ((np.sin(math.pi * (1 + (x[0] + 1) / 4))) ** 2)
            + np.sum(
                (((x[: dim - 1] + 1) / 4) ** 2)
                * (1 + 10 * ((np.sin(math.pi * (1 + (x[1 :] + 1) / 4)))) ** 2)
            )
            + ((x[dim - 1] + 1) / 4) ** 2
        ) + np.sum(Ufun(x, 10, 100, 4))
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions
  


class F13:
    def __init__(self, num_dimensions):
        self.dimensions = num_dimensions
        self.low = -50
        self.high = 50
        self.global_optimum_solution = 0

    def get_func_val(self, x):
        if x.ndim==1:
            x = x.reshape(1,-1)

        o = 0.1 * (
            (np.sin(3 * np.pi * x[:,0])) ** 2
            + np.sum(
                (x[:,:-1] - 1) ** 2
                * (1 + (np.sin(3 * np.pi * x[:,1:])) ** 2), axis=1
            )
            + ((x[:,-1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * x[:,-1])) ** 2)
        ) + np.sum(Ufun(x, 5, 100, 4))
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions




class F14:
    def __init__(self):
        self.dimensions = 2
        self.low = -65.536
        self.high = 65.536
        self.global_optimum_solution = 1

    def get_func_val(self, x):
        aS = [
        [
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
        ],
        [
            -32, -32, -32, -32, -32,
            -16, -16, -16, -16, -16,
            0, 0, 0, 0, 0,
            16, 16, 16, 16, 16,
            32, 32, 32, 32, 32,
        ],
    ]
        aS = np.asarray(aS)
        bS = np.zeros(25)
        v = np.matrix(x)
        for i in range(0, 25):
            H = v - aS[:, i]
            bS[i] = np.sum((np.power(H, 6)))
        w = [i for i in range(25)]
        for i in range(0, 24):
            w[i] = i + 1
        o = ((1.0 / 500) + np.sum(1.0 / (w + bS))) ** (-1)
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions




class F15:
    def __init__(self):
        self.dimensions = 4
        self.low = -5
        self.high = 5
        self.global_optimum_solution = 0.0003075

    def get_func_val(self, L):
        aK = [
            0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
            0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
        ]
        bK = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
        aK = np.asarray(aK)
        bK = np.asarray(bK)
        bK = 1 / bK
        fit = np.sum(
            (aK - ((L[0] * (bK ** 2 + L[1] * bK)) / (bK ** 2 + L[2] * bK + L[3]))) ** 2
        )
        return fit

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions



class F16:
    def __init__(self):
        self.dimensions = 2
        self.low = -5
        self.high = 5
        self.global_optimum_solution = -1.0316285

    def get_func_val(self, L):
        o = (
            4 * (L[0] ** 2)
            - 2.1 * (L[0] ** 4)
            + (L[0] ** 6) / 3
            + L[0] * L[1]
            - 4 * (L[1] ** 2)
            + 4 * (L[1] ** 4)
        )
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions



class F17:
    def __init__(self):
        self.dimensions = 2
        self.low = -5
        self.high = 15
        self.global_optimum_solution = 0.398

    def get_func_val(self, L):
        o = (
            (L[1] - (L[0] ** 2) * 5.1 / (4 * (np.pi ** 2)) + 5 / np.pi * L[0] - 6)
            ** 2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(L[0])
            + 10
        )
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions




class F18:
    def __init__(self):
        self.dimensions = 2
        self.low = -2
        self.high = 2
        self.global_optimum_solution = 3

    def get_func_val(self, L):
        o = (
            1
            + (L[0] + L[1] + 1) ** 2
            * (
                19
                - 14 * L[0]
                + 3 * (L[0] ** 2)
                - 14 * L[1]
                + 6 * L[0] * L[1]
                + 3 * L[1] ** 2
            )
        ) * (
            30
            + (2 * L[0] - 3 * L[1]) ** 2
            * (
                18
                - 32 * L[0]
                + 12 * (L[0] ** 2)
                + 48 * L[1]
                - 36 * L[0] * L[1]
                + 27 * (L[1] ** 2)
            )
        )
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions



class F19:
    def __init__(self):
        self.dimensions = 4
        self.low = 0
        self.high = 1
        self.global_optimum_solution = -3.86

    def get_func_val(self, L):
        aH = [[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]
        aH = np.asarray(aH)
        cH = [1, 1.2, 3, 3.2]
        cH = np.asarray(cH)
        pH = [
            [0.3689, 0.117, 0.2673],
            [0.4699, 0.4387, 0.747],
            [0.1091, 0.8732, 0.5547],
            [0.03815, 0.5743, 0.8828],
        ]
        pH = np.asarray(pH)
        o = 0
        for i in range(0, 4):
            o = o - cH[i] * np.exp(-(np.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions





class F20:
    def __init__(self):
        self.dimensions = 6
        self.low = 0
        self.high = 1
        self.global_optimum_solution = -3.32

    def get_func_val(self, L):
        aH = [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
        aH = np.asarray(aH)
        cH = [1, 1.2, 3, 3.2]
        cH = np.asarray(cH)
        pH = [
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
        ]
        pH = np.asarray(pH)
        o = 0
        for i in range(0, 4):
            o = o - cH[i] * np.exp(-(np.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions


class F21:
    def __init__(self):
        self.dimensions = 4
        self.low = 0
        self.high = 10
        self.global_optimum_solution = -10

    def get_func_val(self, L):
        aSH = [
            [4, 4, 4, 4],
            [1, 1, 1, 1],
            [8, 8, 8, 8],
            [6, 6, 6, 6],
            [3, 7, 3, 7],
            [2, 9, 2, 9],
            [5, 5, 3, 3],
            [8, 1, 8, 1],
            [6, 2, 6, 2],
            [7, 3.6, 7, 3.6],
        ]
        cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
        aSH = np.asarray(aSH)
        cSH = np.asarray(cSH)
        fit = 0
        for i in range(5):
            v = np.matrix(L - aSH[i, :])
            fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
        o = fit.item(0)
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions



class F22:
    def __init__(self):
        self.dimensions = 4
        self.low = 0
        self.high = 10
        self.global_optimum_solution = -10

    def get_func_val(self, L):
        aSH = [
            [4, 4, 4, 4],
            [1, 1, 1, 1],
            [8, 8, 8, 8],
            [6, 6, 6, 6],
            [3, 7, 3, 7],
            [2, 9, 2, 9],
            [5, 5, 3, 3],
            [8, 1, 8, 1],
            [6, 2, 6, 2],
            [7, 3.6, 7, 3.6],
        ]
        cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
        aSH = np.asarray(aSH)
        cSH = np.asarray(cSH)
        fit = 0
        for i in range(7):
            v = np.matrix(L - aSH[i, :])
            fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
        o = fit.item(0)
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions



class F23:
    def __init__(self):
        self.dimensions = 4
        self.low = 0
        self.high = 10
        self.global_optimum_solution = -10

    def get_func_val(self, L):
        aSH = [
            [4, 4, 4, 4],
            [1, 1, 1, 1],
            [8, 8, 8, 8],
            [6, 6, 6, 6],
            [3, 7, 3, 7],
            [2, 9, 2, 9],
            [5, 5, 3, 3],
            [8, 1, 8, 1],
            [6, 2, 6, 2],
            [7, 3.6, 7, 3.6],
        ]
        cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
        aSH = np.asarray(aSH)
        cSH = np.asarray(cSH)
        fit = 0
        for i in range(10):
            v = np.matrix(L - aSH[i, :])
            fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
        o = fit.item(0)
        return o

    def get_search_range(self):
        lows = np.asarray([self.low for i in range(self.dimensions)])
        highs = np.asarray([self.high for i in range(self.dimensions)])
        arrs = [highs, lows]
        return np.asarray(arrs)

    def get_global_optimum_solution(self):
        return self.global_optimum_solution

    def get_dimensions(self):
        return self.dimensions
