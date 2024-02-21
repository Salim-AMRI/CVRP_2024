import logging
import math
import numba as nb
import numpy as np
from numba import cuda, int16, int64, boolean
from numba.cuda.random import xoroshiro128p_uniform_float32

@nb.cuda.jit
def splitting_algorithm(offspring, size_pop, distance_matrix_gpu, client_demands, vehicle_capacity, nb_clients, nb_voitures, rng_states, offsprings_pop, size_route_offspring, demand_route_global_mem, client_affecte):
    d = nb.cuda.grid(1)

    if d < size_pop:

        V = nb.cuda.local.array(size2, nb.int64)
        P = nb.cuda.local.array(size2, nb.int64)

        for i in range(1, size2):
            V[i] = 99999
            P[i] = -1

        V[0] = 0

        for i in range(1, nb_clients + 1):
            demand_route = 0
            cost = 0
            j = i

            while j <= nb_clients and demand_route <= vehicle_capacity[0]:
                if j <= nb_clients:
                    demand_route += client_demands[int(offspring[d, j - 1])]

                if i == j:
                    cost = distance_matrix_gpu[0][int(offspring[d, j - 1])] + distance_matrix_gpu[int(offspring[d, j - 1])][0]
                else:
                    cost = cost - distance_matrix_gpu[int(offspring[d, j - 2])][0] + \
                           distance_matrix_gpu[int(offspring[d, j - 2])][int(offspring[d, j - 1])] + \
                           distance_matrix_gpu[int(offspring[d, j - 1])][0]

                if demand_route <= vehicle_capacity[0]:
                    if V[i - 1] + cost < V[j]:
                        V[j] = V[i - 1] + cost
                        P[j] = i - 1

                    j += 1

        # Initialisation des tournées
        t = 0
        j = nb_clients

        for v in range(nb_voitures):
            offsprings_pop[d, 0, v] = 0

        for c in range(nb_clients):
            client_affecte[c] = 0

        for idx1_v in range(nb_voitures):
            size_route_offspring[d, idx1_v] = 2
            demand_route_global_mem[d, idx1_v] = 0

        i = 1

        # Répétition jusqu'à ce que j atteigne le dépôt (0)
        while i > 0:
            i = P[j]
            cpt = 1
            # Ajouter les clients de i+1 à j à la tournée t
            for k in range(i + 1, j + 1):
                if t < nb_voitures:
                    offsprings_pop[d, cpt, t] = offspring[d, k - 1]
                    client_affecte[offspring[d, k - 1] - 1] = 1
                    size_route_offspring[d, t] += 1
                    demand_route_global_mem[d, t] += client_demands[offspring[d, k - 1]]
                    cpt += 1

            j = i
            t += 1

        for c in range(nb_clients):
            if client_affecte[c] == 0:
                r = int(nb_voitures * xoroshiro128p_uniform_float32(rng_states, d))
                if r > nb_voitures - 1:
                    r = nb_voitures - 1
                offsprings_pop[d, int(size_route_offspring[d, r]) - 1, r] = c + 1
                size_route_offspring[d, r] += 1
                demand_route_global_mem[d, r] += client_demands[c + 1]

        for v in range(nb_voitures):
            offsprings_pop[d, int(size_route_offspring[d, v]) - 1, v] = 0        
