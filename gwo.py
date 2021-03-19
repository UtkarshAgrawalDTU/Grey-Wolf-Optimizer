import numpy as np
from opteval import benchmark_func as bf
import matplotlib.pyplot as plt


def gwo_mean_std(func_name, ndim, n_iter):

    scores = []
    for i in range(30):
        scores.append(gwo(func_name, ndim, n_iter, i))

    best_sol = min(scores)
    return (best_sol, np.mean(scores), np.std(scores))





def gwo(func_name, n_dim, n_iter, index):

    n_pop = 15

    #initializing population
    pop = []
    for i in range(n_pop):
        temp = []
        arrs = func_name.get_search_range()
        for j in range(n_dim):
            high = arrs[0][j]
            low = arrs[1][j]
            val = np.random.uniform(low, high)
            temp.append(val)
        pop.append(temp)
    
    pop = np.asarray(pop)

    pop_x = []
    pop_y = []
    x_alpha_history = []
    fitness_history = []
    convergence_history = []

    #fitness of each wolf
    distance = []
    for i in range(n_pop):
        val = func_name.get_func_val(pop[i])
        distance.append(val)

    distance = np.asarray(distance)

    alpha_index = -1
    beta_index = -1
    delta_index = -1

    for i in range(len(distance)):
        fitness = distance[i]
        
        if alpha_index == -1 or fitness < distance[alpha_index]:
            delta_index = beta_index
            beta_index = alpha_index
            alpha_index = i

        elif beta_index == -1 or fitness < distance[beta_index]:
            delta_index = beta_index
            beta_index = i

        elif delta_index == -1 or fitness < distance[delta_index]:
            delta_index = i

    x_alpha = pop[alpha_index]
    x_beta = pop[beta_index]
    x_delta = pop[delta_index]

    a = 2
    r1 = np.random.rand()
    r2 = np.random.rand()
    C = []
    A = []

    for i in range(3):
        r1 = np.random.rand(n_dim)
        r2 = np.random.rand(n_dim)
        A.append(2 * r1 * a - a)
        C.append(2 * r2)

    C = np.asarray(C)
    A = np.asarray(A)


    best_alpha_score = func_name.get_func_val(x_alpha)

    for x in range(n_iter):
        
        #updating coords of each wolf
        for j in range(n_pop):

            D_alpha = np.abs(C[0] * x_alpha - pop[j])
            D_beta = np.abs(C[1] * x_beta - pop[j])
            D_delta = np.abs(C[2] * x_delta - pop[j])

            X1 = x_alpha - A[0] * D_alpha
            X2 = x_beta - A[1] * D_beta
            X3 = x_delta - A[2] * D_delta

            X_new = (X1 + X2 + X3)/3
            
            arrs = func_name.get_search_range()
            high = arrs[0]
            low = arrs[1]

            flag_ub = []
            for i in range(len(X_new)):
                if X_new[i] > high[i]:
                    flag_ub.append(True)
                else:
                    flag_ub.append(False)
            flag_ub = np.asarray(flag_ub)

            flag_lb = []
            for i in range(len(X_new)):
                if X_new[i] < low[i]:
                    flag_lb.append(True)
                else:
                    flag_lb.append(False)
            flag_lb = np.asarray(flag_lb)

            X_new = np.multiply(X_new, np.logical_not(np.logical_or(flag_lb, flag_ub))) + np.multiply(high, flag_ub) + np.multiply(low, flag_lb)
            pop[j] = X_new
        

        for i in range(len(pop)):
            pop_x.append(pop[i][0])
            pop_y.append(pop[i][1])


        #updating constants
        a = 2*(1 - x/n_iter)
        C = []
        A = []

        for i in range(3):
            r1 = np.random.rand()
            r2 = np.random.rand()
            A.append(2 * r1 * a - a)
            C.append(2 * r2) 

        C = np.asarray(C)
        A = np.asarray(A)
        

        #fitness
        distance = []
        for i in range(n_pop):
            val = func_name.get_func_val(pop[i])
            distance.append(val)

        distance = np.asarray(distance)

        alpha_index = -1
        beta_index =  -1   
        delta_index = -1

        #finding best wolves
        for i in range(len(distance)):
            fitness = distance[i]

            if alpha_index == -1 or fitness < distance[alpha_index]:
                delta_index = beta_index
                beta_index = alpha_index
                alpha_index = i

            elif beta_index == -1 or fitness < distance[beta_index]:
                delta_index = beta_index
                beta_index = i

            elif delta_index == -1 or fitness < distance[delta_index]:
                delta_index = i

        x_alpha = pop[alpha_index]
        x_beta = pop[beta_index]
        x_delta = pop[delta_index]
        best_alpha_score = min(best_alpha_score, func_name.get_func_val(x_alpha))

        x_alpha_history.append(pop[0][0])
        fitness_history.append(func_name.get_func_val(x_alpha))
        convergence_history.append(best_alpha_score)

    
    if index == 0:
        x_alpha_history = np.asarray(x_alpha_history)
        fitness_history = np.asarray(fitness_history)
        convergence_history = np.asarray(convergence_history)

        f, (a0, a1, a2, a3) = plt.subplots(nrows = 1, ncols = 4, figsize = (15, 4))
        f.tight_layout()
        
        a0.scatter(pop_x, pop_y)
        a0.set_xlim(arrs[1][0], arrs[0][0])
        a0.set_ylim(arrs[1][1], arrs[0][1])
        a0.set_title('Search History')
        a0.set_xlabel('x0')

        a1.plot(x_alpha_history, color = 'r')
        a1.set_title('Trajectory in First Dimension')
        a1.set_xlabel('Iterations')


        a2.plot(fitness_history, color = 'b')
        a2.set_title('Fitness History')
        a2.set_xlabel('Iterations')

        a3.plot(convergence_history, color = 'g')
        a3.set_title('Convergence Curve')
        a3.set_xlabel('Iterations')

        plt.show()

    return best_alpha_score
    
