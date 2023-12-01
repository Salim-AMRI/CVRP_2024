import logging
import math

import numba as nb
import numpy as np
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

nb_clients = -1
nb_voitures = -1
size = -1

max_size_route = -1


#@cuda.jit
#def computeClosestCrossover_OX(rng_states, size_pop, distance_matrix_gpu, solution_pop, offspring, size_route_offspring, demand_route_global_mem, client_demands, indices):


    #d = cuda.grid(1)
    #nbParent = 2

    #if d < size_pop:

        #idx1 = int(d)
        #idx2 = int(indices[idx1])
        
        #parents = nb.cuda.local.array((nbParent, max_size_route, nb_voitures), nb.int16)

        #for idx1_v in range(nb_voitures):
            #for idx1_c in range(max_size_route):
                #parents[0, idx1_c, idx1_v] = solution_pop[idx1, idx1_c, idx1_v]
                #parents[1, idx1_c, idx1_v] = solution_pop[idx2, idx1_c, idx1_v]
        
        
        #.......
        
        #offspring[d, 0, 0] = 0
        #offspring[d, 1, 0] = client1
        #offspring[d, 2, 0] = client2
        #offspring[d, 3, 0] = 0
        #offspring[d, 4, 0] = -1
        
@cuda.jit
def computeClosestCrossover_GPX(rng_states, size_pop, distance_matrix_gpu, solution_pop, offspring, size_route_offspring, demand_route_global_mem, client_demands, indices):


    d = cuda.grid(1)
    nbParent = 2

    if d < size_pop:

        idx1 = int(d)
        idx2 = int(indices[idx1])

        parents = nb.cuda.local.array((nbParent, max_size_route, nb_voitures), nb.int16)

        for idx1_v in range(nb_voitures):
            for idx1_c in range(max_size_route):
                parents[0, idx1_c, idx1_v] = solution_pop[idx1, idx1_c, idx1_v]
                parents[1, idx1_c, idx1_v] = solution_pop[idx2, idx1_c, idx1_v]


        for idx1_v in range(nb_voitures):
            size_route_offspring[d, idx1_v] = 2
            demand_route_global_mem[d, idx1_v] = 0


        distances_routes = nb.cuda.local.array((nbParent, nb_voitures), nb.float32)
        quality_routes = nb.cuda.local.array((nbParent, nb_voitures), nb.float32)

        client_affecte = nb.cuda.local.array((nb_clients), nb.int16)
        
        for c in range(nb_clients):
            client_affecte[c] = 0


        position_client_parents = nb.cuda.local.array((nbParent, nb_clients, 2), nb.int16)
        
        

        for i in range(nbParent):

            for v in range(nb_voitures):

                distances_routes[i, v] = 0
                quality_routes[i, v] = 0

                for c in range(max_size_route - 1):

                    if(parents[i, c + 1, v] != -1):

                        client = parents[i, c, v]
                        next_client =  parents[i, c + 1, v]
                        
                        distances_routes[i, v] +=  distance_matrix_gpu[client, next_client]

                        if ( parents[i, c, v] != 0):

                            quality_routes[i, v] += distance_matrix_gpu[0, client]
                            
                        if(client != 0 and client != -1):
                            
                            position_client_parents[i, client - 1, 0] = c
                            position_client_parents[i, client - 1, 1] = v  
                            
                            

        for i in range(nb_voitures):

            indiceParent = i % 2
            indiceOtherparent = 1 - indiceParent

            valMax = -1
            voitureMax = -1

            for j in range(nb_voitures):

                if(quality_routes[indiceParent, j] > 0 and distances_routes[indiceParent, j] > 0):
                    
                    currentVal = quality_routes[indiceParent, j]/distances_routes[indiceParent, j]

                    if currentVal > valMax:
                        valMax = currentVal
                        voitureMax = j

            cpt = 0

            for c in range(max_size_route):

                
                client = parents[int(indiceParent), c, int(voitureMax)]
                offspring[d, cpt, i] = client
                
                cpt = cpt + 1


                if (client != -1 and client != 0):
                    
                    client_affecte[client - 1] = 1

                    size_route_offspring[d, i] += 1
                    demand_route_global_mem[d,i] += client_demands[client]
                    
                    
                    parents[int(indiceOtherparent), int(position_client_parents[int(indiceOtherparent), client - 1, 0]), int(position_client_parents[int(indiceOtherparent), client - 1, 1])] = -1




            quality_routes[int(indiceParent), voitureMax] = -999
            distances_routes[int(indiceParent), voitureMax] = 1


            ### A améliorer, ne pas parcourir route déjà utilisées

            for v in range(nb_voitures):

                cpt2 = 1
                
                if(quality_routes[int(indiceOtherparent), v] != -999):

                    distances_routes[indiceOtherparent, v] = 0
                    quality_routes[indiceOtherparent, v] = 0


                    for c in range(max_size_route):

                        if(parents[indiceOtherparent, c , v] != -1):

                            if (parents[indiceOtherparent, c, v] != 0):
                                
                                quality_routes[indiceOtherparent, v] += distance_matrix_gpu[0, parents[indiceOtherparent, c , v]]

                            if(c == 0):
                                client = parents[indiceOtherparent, c , v]
                            else:
                                next_client = parents[indiceOtherparent, c, v]
                                distances_routes[indiceOtherparent, v] += distance_matrix_gpu[client, next_client]
                                
                                parents[indiceOtherparent, cpt2, v] = next_client
                                
                                if(next_client != 0 and next_client != -1):
                                    position_client_parents[indiceOtherparent, next_client - 1, 0] = cpt2
                            
                                cpt2 += 1
                                
                                client = next_client

                    for c in range(cpt2, max_size_route):
                        parents[indiceOtherparent, c, v] = -1
                
                
  

        for c in range(nb_clients):
            if(client_affecte[c] == 0):
                r = int(nb_voitures * xoroshiro128p_uniform_float32(rng_states, d))
                
                if(r > nb_voitures - 1):
                    r = nb_voitures - 1
                    
                offspring[d, int(size_route_offspring[d, r]) - 1,r] = c + 1
                
                size_route_offspring[d, r] += 1
                
                demand_route_global_mem[d,r] += client_demands[c + 1]
        
        for v in range(nb_voitures):
            offspring[d,int(size_route_offspring[d, v]) - 1, v] = 0

