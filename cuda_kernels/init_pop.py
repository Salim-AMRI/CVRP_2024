from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32
import numba as nb

nb_clients = -1
nb_voitures = -1



#################
@cuda.jit
def generate_illegal_solutions(D, rng_states, demand_route, client_demands, current_solution,
                                size_route):
    d = cuda.grid(1)

    if d < D:


        for v in range(nb_voitures):
            
            for c in range( nb_clients + 2):
                current_solution[d, c, v] = -1

        size_route_local = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        for idx1_v in range(nb_voitures):
            size_route_local[idx1_v] = 1
            
            

        for v in range(nb_voitures):
            current_solution[d, 0, v] = 0
            #size_route[d, v] = 1
            

        for i in range(1, nb_clients + 1):


                
            r = int(nb_voitures * xoroshiro128p_uniform_float32(rng_states, d))
            
            if(r > nb_voitures - 1):
                r = nb_voitures - 1
            if(r < 0):
                r = 0

            #r = i % nb_voitures

            #if(i == 55):
                #test2[d] = int(size_route_local[0])
                

                
                
            demand_route[d, r] += client_demands[i]
            
                
            current_solution[d, int(size_route_local[r]), r] = int(i)
                
            size_route_local[r]  =  size_route_local[r] + 1

                

                
                
        for v in range(nb_voitures):
           current_solution[d, int(size_route_local[v]), v] = 0
           size_route_local[v] += 1
            
        for idx1_v in range(nb_voitures):
            size_route[d, idx1_v] = size_route_local[idx1_v]
            
        
        


#################
@cuda.jit
def generate_legal_solutions(D, rng_states, demand_route, client_demands, vehicle_capacity, current_solution, size_route):
    
    d = cuda.grid(1)

    if d < D:

        for v in range(nb_voitures):
            demand_route[d, v] = 0
        

        for i in range(1, nb_clients + 1):
            
            flag = 1
            
            while flag == 1:

                r = int(nb_voitures * xoroshiro128p_uniform_float32(rng_states, d))
                
                if (demand_route[d, r] + client_demands[i] <= vehicle_capacity[r]):
                    demand_route[d,r] += client_demands[i]
                    current_solution[d,size_route[d,r], r] = i
                    size_route[d,r] += 1
                    flag  = 0

        for v in range(nb_voitures):
            current_solution[d, size_route[d, v] ,v ] = 0
            size_route[d, v] += 1
            
        
        
#################
@cuda.jit
def repair_solutions(D, rng_states, distance_matrix_gpu, demand_route, client_demands, vehicle_capacity, current_solution, size_route_global_mem):

    d = cuda.grid(1)

    if d < D:
        
        
        f = 0

        for idx1_v in range(nb_voitures):
            for idx1_c in range(int(size_route_global_mem[d,idx1_v]-1)):
                f += int(distance_matrix_gpu[current_solution[d, idx1_c, idx1_v],current_solution[d,idx1_c + 1, idx1_v]])
                
                
        for idx1_v in range(nb_voitures):
            
            
            
            for idx1_c in range(1, int(size_route_global_mem[d,idx1_v] - 1)):
                
                if(demand_route[d,idx1_v] > vehicle_capacity[idx1_v]):
                    
                    # Leaving client x
                    current_client = current_solution[d,idx1_c, idx1_v]
                    
                    client_before =current_solution[d, idx1_c - 1, idx1_v]
                    client_after =current_solution[d, idx1_c + 1, idx1_v]
                    impact_client_leaving = distance_matrix_gpu[client_before, client_after] - distance_matrix_gpu[current_client, client_before] - distance_matrix_gpu[current_client, client_after]

                    
                    best_delta = 99999
                    best_idx1_client = -1
                    best_idx1_voiture = -1
                    best_idx2_client = -1
                    best_idx2_voiture = -1
                    
                    for idx2_v in range(nb_voitures):
                        
                        if idx1_v != idx2_v and demand_route[d, idx2_v] + client_demands[current_client] <= vehicle_capacity[idx2_v]:
                            
                            for idx2_c in range(int(size_route_global_mem[d,idx2_v] - 1)):
                                
                                # new position for client x
                                new_client_before =current_solution[d, idx2_c, idx2_v]

                                if new_client_before != current_client and new_client_before != client_before:

                                    new_client_after =current_solution[d, idx2_c + 1, idx2_v]
                                    impact_client_arrival = distance_matrix_gpu[new_client_before, current_client] + distance_matrix_gpu[current_client, new_client_after] - distance_matrix_gpu[new_client_before, new_client_after]

                                    delta = int(impact_client_leaving + impact_client_arrival)
                
                
                                    if(delta < best_delta):
                            
                                        best_delta = delta
                                        best_idx1_client = idx1_c
                                        best_idx1_voiture = idx1_v
                                        best_idx2_client = idx2_c
                                        best_idx2_voiture = idx2_v            
        


                    f += int(best_delta)

                    # Mise à jour de la current solution
                    best_client =current_solution[d, best_idx1_client, best_idx1_voiture]


                    
                    # Mise à jour de la route 1
                    for idx1_c in range(best_idx1_client, size_route_global_mem[d, best_idx1_voiture]):
                        current_solution[d, idx1_c, best_idx1_voiture] =current_solution[d, idx1_c + 1, best_idx1_voiture]

                    if best_idx1_voiture == best_idx2_voiture and best_idx2_client > best_idx1_client:
                        best_idx2_client = best_idx2_client - 1

                    # Mise à jour de la route 2
                    for i in range(0, size_route_global_mem[d,best_idx2_voiture] - best_idx2_client):
                        idx2_c = size_route_global_mem[d, best_idx2_voiture] - i
                        current_solution[d,idx2_c + 1, best_idx2_voiture] = current_solution[d, idx2_c, best_idx2_voiture]


                    current_solution[d, best_idx2_client + 1, best_idx2_voiture] = best_client



                    # Mise à jour de la size route
                    size_route_global_mem[d, best_idx1_voiture] = size_route_global_mem[d, best_idx1_voiture] - 1
                    size_route_global_mem[d, best_idx2_voiture] = size_route_global_mem[d, best_idx2_voiture] + 1

                    # Mise à jour de la capacite
                    demand_route[d, best_idx1_voiture] = demand_route[d, best_idx1_voiture] -  client_demands[best_client]
                    demand_route[d, best_idx2_voiture] = demand_route[d, best_idx2_voiture] +  client_demands[best_client]
                                                                                                                                    
                                                                                                                                    
