import numba as nb
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32


nb_clients = -1
nb_voitures = -1
size = -1
max_size_route = -1
nb_nearest_neighbor_tabu = -1
infinit = 99999


@cuda.jit
def tabu_CVRP_legal(rng_states, D, max_iter, distance_matrix_gpu, current_solution_global_mem, demand_route_global_mem, vehicle_capacity_global_mem, client_demands_global_mem, size_route_global_mem, vector_f_global_mem, alpha):
    # Définit le kernel CUDA pour l'algorithme de recherche tabou pour le problème de routage de véhicules avec capacité (CVRP).

    d = cuda.grid(1)

    if d < D:

        tabuTenure = nb.cuda.local.array((nb_clients), dtype=nb.int16)

        for x in range(nb_clients ):
            tabuTenure[x] = 0
            

        current_solution_local = nb.cuda.local.array((size, nb_voitures), dtype=nb.int16)
        for idx1_v in range(nb_voitures):
            for idx1_c in range(nb_clients + 2):
                current_solution_local[idx1_c, idx1_v] = current_solution_global_mem[d, idx1_c, idx1_v]
                

        current_demande_local = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        for idx1_v in range(nb_voitures):
            current_demande_local[idx1_v] = demand_route_global_mem[d, idx1_v]
            
            
        f = 0

        for idx1_v in range(nb_voitures):
            for idx1_c in range(int(size_route_global_mem[d,idx1_v]-1)):
                
                f += int(distance_matrix_gpu[current_solution_local[idx1_c, idx1_v],current_solution_local[idx1_c + 1, idx1_v]])
            
        
        
        f_best = f
        
        
        for iter_ in range(max_iter):

            best_delta = 99999
            best_idx1_client = -1
            best_idx1_voiture = -1
            best_idx2_client = -1
            best_idx2_voiture = -1

            for idx1_v in range(nb_voitures):
                for idx1_c in range(1, int(size_route_global_mem[d,idx1_v] - 1)):
                    # Leaving client x
                    current_client = current_solution_local[idx1_c, idx1_v]
                    
                    
                    
                    client_before =current_solution_local[ idx1_c - 1, idx1_v]
                    client_after =current_solution_local[ idx1_c + 1, idx1_v]
                    impact_client_leaving = distance_matrix_gpu[client_before, client_after] - distance_matrix_gpu[current_client, client_before] - distance_matrix_gpu[current_client, client_after]

                    
                    for idx2_v in range(nb_voitures):
                        if idx1_v == idx2_v or (current_demande_local[idx2_v] + client_demands_global_mem[current_client] <= vehicle_capacity_global_mem[idx2_v]):
                            for idx2_c in range(int(size_route_global_mem[d,idx2_v] - 1)):
                                # new position for client x
                                new_client_before =current_solution_local[ idx2_c, idx2_v]

                                if new_client_before != current_client and new_client_before != client_before:

                                    new_client_after =current_solution_local[ idx2_c + 1, idx2_v]
                                    impact_client_arrival = distance_matrix_gpu[new_client_before, current_client] + distance_matrix_gpu[current_client, new_client_after] - distance_matrix_gpu[new_client_before, new_client_after]

                                    delta = int(impact_client_leaving + impact_client_arrival)
                                    


    
                                    #if (tabuTenure[current_client] <= iter_  or delta + f < X_best):
                                    if (tabuTenure[current_client-1] <= iter_  or delta + f < f_best ):    
                                        
                                        if(delta < best_delta):
                                
                                            best_delta = delta
                                            best_idx1_client = idx1_c
                                            best_idx1_voiture = idx1_v
                                            best_idx2_client = idx2_c
                                            best_idx2_voiture = idx2_v

            f += int(best_delta)

            # Mise à jour de la current solution
            best_client =current_solution_local[ best_idx1_client, best_idx1_voiture]


            #test =current_solution_local[ best_idx2_client, best_idx2_voiture]
            #test = best_idx2_voiture

            
            # Mise à jour de la route 1
            for idx1_c in range(best_idx1_client, size_route_global_mem[d, best_idx1_voiture]):
               current_solution_local[ idx1_c, best_idx1_voiture] =current_solution_local[ idx1_c + 1, best_idx1_voiture]

            if best_idx1_voiture == best_idx2_voiture and best_idx2_client > best_idx1_client:
                best_idx2_client = best_idx2_client - 1

            # Mise à jour de la route 2
            for i in range(0, size_route_global_mem[d,best_idx2_voiture] - best_idx2_client):
                idx2_c = size_route_global_mem[d, best_idx2_voiture] - i
                current_solution_local[idx2_c + 1, best_idx2_voiture] = current_solution_local[ idx2_c, best_idx2_voiture]


            current_solution_local[ best_idx2_client + 1, best_idx2_voiture] = best_client

            list_tabu = int(alpha * nb_clients + int((10 * xoroshiro128p_uniform_float32(rng_states, d)) + 1))

            tabuTenure[best_client-1] = iter_ + list_tabu

            # Mise à jour de la size route
            size_route_global_mem[d, best_idx1_voiture] = size_route_global_mem[d, best_idx1_voiture] - 1
            size_route_global_mem[d, best_idx2_voiture] = size_route_global_mem[d, best_idx2_voiture] + 1

            # Mise à jour de la capacite
            current_demande_local[ best_idx1_voiture] = current_demande_local[ best_idx1_voiture] -  client_demands_global_mem[best_client]
            current_demande_local[ best_idx2_voiture] = current_demande_local[best_idx2_voiture] +  client_demands_global_mem[best_client]

            # Mise à jour de la fitness et des structures en fonction du meilleur mouvement.
            if f < f_best:
                
                f_best = f
                # Mise à jour de la meilleure solution (best_solution)

                for idx1_v in range(nb_voitures):
                    for idx1_c in range(nb_clients + 2):
                       current_solution_global_mem[d, idx1_c, idx1_v] =current_solution_local[ idx1_c, idx1_v]
                    
        
        vector_f_global_mem[d] = f_best
        
        
        
        
@cuda.jit
def tabu_CVRP_lambda(rng_states, D, max_iter, distance_matrix_gpu, current_solution_global_mem,
                      demand_route_global_mem, vehicle_capacity_global_mem, client_demands_global_mem,
                      size_route_global_mem,  vector_f_global_mem, lambda_, alpha, 
                      nb_iteration):
    d = cuda.grid(1)

    if d < D:

        tabuTenure = nb.cuda.local.array((nb_clients), dtype=nb.int16)

        current_solution_local = nb.cuda.local.array((size, nb_voitures), dtype=nb.int16)
        for idx1_v in range(nb_voitures):
            for idx1_c in range(nb_clients + 2):
                current_solution_local[idx1_c, idx1_v] = current_solution_global_mem[d, idx1_c, idx1_v]

        best_iteration_solution_local = nb.cuda.local.array((size, nb_voitures), dtype=nb.int16)

        for idx1_v in range(nb_voitures):
            for idx1_c in range(nb_clients + 2):
                best_iteration_solution_local[idx1_c, idx1_v] = current_solution_local[idx1_c, idx1_v]

        current_demande_local = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        for idx1_v in range(nb_voitures):
            current_demande_local[idx1_v] = demand_route_global_mem[d, idx1_v]

        best_iteration_demande_local = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        for idx1_v in range(nb_voitures):
            best_iteration_demande_local[idx1_v] = current_demande_local[idx1_v]

        f_best_legal = 99999

        ##########

        for _ in range(nb_iteration):

            for x in range(nb_clients):
                tabuTenure[x] = 0

            score_distance = 0

            total_penalty = 0

            for idx1_v in range(nb_voitures):

                route_demand = 0

                for idx1_c in range(int(size_route_global_mem[d, idx1_v] - 1)):
                    client = current_solution_local[idx1_c, idx1_v]
                    next_client = current_solution_local[idx1_c + 1, idx1_v]

                    score_distance += distance_matrix_gpu[client, next_client]
                    route_demand += client_demands_global_mem[client]

                if (route_demand > vehicle_capacity_global_mem[idx1_v]):
                    capacity_violation = route_demand - vehicle_capacity_global_mem[idx1_v]
                else:
                    capacity_violation = 0

                total_penalty += lambda_ * capacity_violation

            f = score_distance + total_penalty

            f_best = f

            ####### Incremental Moves

            for iter_ in range(max_iter):

                best_delta = 99999
                best_idx1_client = -1
                best_idx1_voiture = -1
                best_idx2_client = -1
                best_idx2_voiture = -1

                for idx1_v in range(nb_voitures):
                    for idx1_c in range(1, int(size_route_global_mem[d, idx1_v] - 1)):
                        
                        
                        # Leaving client x
                        current_client = current_solution_local[idx1_c, idx1_v]
                        
                        if (tabuTenure[current_client - 1] <= iter_):
                                
                            client_before = current_solution_local[idx1_c - 1, idx1_v]
                            client_after = current_solution_local[idx1_c + 1, idx1_v]
                            impact_client_leaving = distance_matrix_gpu[client_before, client_after] - distance_matrix_gpu[
                                current_client, client_before] - distance_matrix_gpu[current_client, client_after]

                            # Impact de la capacité en quittant le client
                            if vehicle_capacity_global_mem[idx1_v] > current_demande_local[idx1_v] - \
                                    client_demands_global_mem[current_client]:
                                demand_before_leaving = vehicle_capacity_global_mem[idx1_v]
                            else:
                                demand_before_leaving = current_demande_local[idx1_v] - client_demands_global_mem[
                                    current_client]
                                
                                
                            if current_demande_local[idx1_v] > demand_before_leaving:
                                impact_capacite_client_leaving = current_demande_local[idx1_v] - demand_before_leaving
                            else:
                                impact_capacite_client_leaving = 0

                            for idx2_v in range(nb_voitures):
                                for idx2_c in range(int(size_route_global_mem[d, idx2_v] - 1)):
                                    # new position for client x
                                    new_client_before = current_solution_local[idx2_c, idx2_v]

                                    if new_client_before != current_client and new_client_before != client_before:

                                        new_client_after = current_solution_local[idx2_c + 1, idx2_v]
                                        impact_client_arrival = distance_matrix_gpu[new_client_before, current_client] + \
                                                                distance_matrix_gpu[current_client, new_client_after] - \
                                                                distance_matrix_gpu[new_client_before, new_client_after]



                                        delta = impact_client_leaving + impact_client_arrival
                                        # test = delta

                                        # Ajout de la pénalité liée à la capacité dans delta
                                        if (idx1_v != idx2_v):
                                            
                                            # Impact de la capacité en arrivant au nouveau client
                                            if vehicle_capacity_global_mem[idx2_v] > current_demande_local[idx2_v]:
                                                demand_after_arrival = vehicle_capacity_global_mem[idx2_v]
                                            else:
                                                demand_after_arrival = current_demande_local[idx2_v]

                                            if current_demande_local[idx2_v] + client_demands_global_mem[
                                                current_client] > demand_after_arrival:
                                                impact_capacite_client_arrival = current_demande_local[idx2_v] + \
                                                                                client_demands_global_mem[
                                                                                    current_client] - demand_after_arrival
                                            else:
                                                impact_capacite_client_arrival = 0
                                            
                                            
                                            delta_capacity = impact_capacite_client_arrival - impact_capacite_client_leaving
                                            delta_penalty = lambda_ * delta_capacity
                                            delta += delta_penalty
                                            # test = delta_penalty
                                        # test = delta

                                        # if (tabuTenure[current_client] <= iter_ or delta + f < X_best):
                                        #if ((tabuTenure[current_client - 1] <= iter_) or ((delta + f) < f_best)) and (
                                        if(delta < best_delta):
                                            best_delta = delta
                                            best_idx1_client = idx1_c
                                            best_idx1_voiture = idx1_v
                                            best_idx2_client = idx2_c
                                            best_idx2_voiture = idx2_v
                                        # test = best_idx2_voiture


                f += best_delta
                # test = f

                # Mise à jour de la current solution
                best_client = current_solution_local[best_idx1_client, best_idx1_voiture]

                if best_client != 0 and best_client != -1:
                    # Mise à jour de la route 1
                    for idx1_c in range(best_idx1_client, size_route_global_mem[d, best_idx1_voiture]):
                        current_solution_local[idx1_c, best_idx1_voiture] = current_solution_local[
                            idx1_c + 1, best_idx1_voiture]

                    if best_idx1_voiture == best_idx2_voiture and best_idx2_client > best_idx1_client:
                        best_idx2_client = best_idx2_client - 1

                    # Mise à jour de la route 2
                    for i in range(0, size_route_global_mem[d, best_idx2_voiture] - best_idx2_client):
                        idx2_c = size_route_global_mem[d, best_idx2_voiture] - i
                        current_solution_local[idx2_c + 1, best_idx2_voiture] = current_solution_local[
                            idx2_c, best_idx2_voiture]

                    current_solution_local[best_idx2_client + 1, best_idx2_voiture] = best_client

                    list_tabu = int(alpha * nb_clients + int((10 * xoroshiro128p_uniform_float32(rng_states, d)) + 1))

                    tabuTenure[best_client - 1] = iter_ + list_tabu

                    # Mise à jour de la size route
                    size_route_global_mem[d, best_idx1_voiture] -= 1
                    size_route_global_mem[d, best_idx2_voiture] += 1

                    # Mise à jour de la capacite
                    current_demande_local[best_idx1_voiture] -= client_demands_global_mem[best_client]
                    current_demande_local[best_idx2_voiture] += client_demands_global_mem[best_client]

                # Mise à jour de la fitness et des structures en fonction du meilleur mouvement.
                if f < f_best:

                    f_best = f
                    # Mise à jour de la meilleure solution (best_solution)

                    for idx1_v in range(nb_voitures):
                        for idx1_c in range(nb_clients + 2):
                            best_iteration_solution_local[idx1_c, idx1_v] = current_solution_local[idx1_c, idx1_v]
                    #######
                    for idx1_v in range(nb_voitures):
                        best_iteration_demande_local[idx1_v] = current_demande_local[idx1_v]

                    islegal = True
                    for idx1_v in range(nb_voitures):
                        if (current_demande_local[idx1_v] > vehicle_capacity_global_mem[idx1_v]):
                            islegal = False


                    if (islegal and f_best < f_best_legal):
                        
                        f_best_legal = f_best

                        for idx1_v in range(nb_voitures):
                            for idx1_c in range(nb_clients + 2):
                                current_solution_global_mem[d, idx1_c, idx1_v] = best_iteration_solution_local[idx1_c, idx1_v]

                # test = f_best

            # Ajustement de lambda_penalty en fonction de la demande finale

            islegal = True
            for idx1_v in range(nb_voitures):
                if best_iteration_demande_local[idx1_v] > vehicle_capacity_global_mem[idx1_v]:
                    islegal = False

                                
                                
            if islegal:
                
                lambda_ = lambda_//2
                if(lambda_ < 1):
                   lambda_ = 1
                
            else:
                lambda_ = lambda_ * 2

            for idx1_v in range(nb_voitures):
                for idx1_c in range(nb_clients + 2):
                    current_solution_local[idx1_c, idx1_v] = best_iteration_solution_local[idx1_c, idx1_v]

            for idx1_v in range(nb_voitures):
                current_demande_local[idx1_v] = best_iteration_demande_local[idx1_v]

            for idx1_v in range(nb_voitures):
                size_route_global_mem[d, idx1_v] = 0

            for idx1_v in range(nb_voitures):
                for idx1_c in range(nb_clients + 2):
                    if current_solution_local[idx1_c, idx1_v] != -1:
                        size_route_global_mem[d, idx1_v] += 1

        vector_f_global_mem[d] = f_best_legal









@cuda.jit
def tabu_CVRP_lambda_swap(rng_states, D, max_iter, distance_matrix_gpu, current_solution_global_mem,
                      demand_route_global_mem, vehicle_capacity_global_mem, client_demands_global_mem,
                      size_route_global_mem,  vector_f_global_mem, lambda_, alpha, 
                      nb_iteration):
    d = cuda.grid(1)

    if d < D:

        tabuTenure = nb.cuda.local.array((nb_clients), dtype=nb.int16)


        current_solution_local = nb.cuda.local.array((size, nb_voitures), dtype=nb.int16)
        for idx1_v in range(nb_voitures):
            for idx1_c in range(nb_clients + 2):
                current_solution_local[idx1_c, idx1_v] = current_solution_global_mem[d, idx1_c, idx1_v]

        best_iteration_solution_local = nb.cuda.local.array((size, nb_voitures), dtype=nb.int16)

        for idx1_v in range(nb_voitures):
            for idx1_c in range(nb_clients + 2):
                best_iteration_solution_local[idx1_c, idx1_v] = current_solution_local[idx1_c, idx1_v]

        current_demande_local = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        for idx1_v in range(nb_voitures):
            current_demande_local[idx1_v] = demand_route_global_mem[d, idx1_v]

        best_iteration_demande_local = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        for idx1_v in range(nb_voitures):
            best_iteration_demande_local[idx1_v] = current_demande_local[idx1_v]

        f_best_legal = 99999

        ##########

        for _ in range(nb_iteration):

            for x in range(nb_clients):
                tabuTenure[x] = 0

            score_distance = 0

            total_penalty = 0

            for idx1_v in range(nb_voitures):

                route_demand = 0

                for idx1_c in range(int(size_route_global_mem[d, idx1_v] - 1)):
                    client = current_solution_local[idx1_c, idx1_v]
                    next_client = current_solution_local[idx1_c + 1, idx1_v]

                    score_distance += distance_matrix_gpu[client, next_client]
                    route_demand += client_demands_global_mem[client]

                if (route_demand > vehicle_capacity_global_mem[idx1_v]):
                    capacity_violation = route_demand - vehicle_capacity_global_mem[idx1_v]
                else:
                    capacity_violation = 0

                total_penalty += lambda_ * capacity_violation

            f = score_distance + total_penalty

            f_best = f

            ####### Incremental Moves

            for iter_ in range(max_iter):

                best_delta = 99999
                best_idx1_client = -1
                best_idx1_voiture = -1
                best_idx2_client = -1
                best_idx2_voiture = -1
                best_is_swap = 0

                for idx1_v in range(nb_voitures):
                    for idx1_c in range(1, int(size_route_global_mem[d, idx1_v] - 1)):
                        # Leaving client x
                        current_client = current_solution_local[idx1_c, idx1_v]
                        
                        if (tabuTenure[current_client - 1] <= iter_):
                            
                            client_before = current_solution_local[idx1_c - 1, idx1_v]
                            client_after = current_solution_local[idx1_c + 1, idx1_v]
                            impact_client_leaving = distance_matrix_gpu[client_before, client_after] - distance_matrix_gpu[
                                current_client, client_before] - distance_matrix_gpu[current_client, client_after]

                            # Impact de la capacité en quittant le client
                            if vehicle_capacity_global_mem[idx1_v] > current_demande_local[idx1_v] - \
                                    client_demands_global_mem[current_client]:
                                demand_before_leaving = vehicle_capacity_global_mem[idx1_v]
                            else:
                                demand_before_leaving = current_demande_local[idx1_v] - client_demands_global_mem[
                                    current_client]
                                
                                
                            if current_demande_local[idx1_v] > demand_before_leaving:
                                impact_capacite_client_leaving = current_demande_local[idx1_v] - demand_before_leaving
                            else:
                                impact_capacite_client_leaving = 0


                            for idx2_v in range(nb_voitures):
                                for idx2_c in range(int(size_route_global_mem[d, idx2_v] - 1)):
                                    # new position for client x
                                    new_client_before = current_solution_local[idx2_c, idx2_v]


                                    ### Eval relocate move
                                    if new_client_before != current_client and (new_client_before != client_before or (idx2_c ==0 and idx1_v != idx2_v)):

                                        new_client_after = current_solution_local[idx2_c + 1, idx2_v]
                                        impact_distance_client_arrival = distance_matrix_gpu[new_client_before, current_client] + \
                                                                distance_matrix_gpu[current_client, new_client_after] - \
                                                                distance_matrix_gpu[new_client_before, new_client_after]



                                        delta = impact_client_leaving + impact_distance_client_arrival
                                        # test = delta

                                        # Ajout de la pénalité liée à la capacité dans delta
                                        if (idx1_v != idx2_v):
                                            
                                            # Impact de la capacité en arrivant au nouveau client
                                            if vehicle_capacity_global_mem[idx2_v] > current_demande_local[idx2_v]:
                                                demand_after_arrival = vehicle_capacity_global_mem[idx2_v]
                                            else:
                                                demand_after_arrival = current_demande_local[idx2_v]

                                            if current_demande_local[idx2_v] + client_demands_global_mem[current_client] > demand_after_arrival:
                                                
                                                impact_capacite_client_arrival = current_demande_local[idx2_v] + \
                                                                                client_demands_global_mem[
                                                                                    current_client] - demand_after_arrival
                                            else:
                                                impact_capacite_client_arrival = 0
                                            
                                            delta_capacity = impact_capacite_client_arrival - impact_capacite_client_leaving
                                            delta_penalty = lambda_ * delta_capacity
                                            delta += delta_penalty
                                            # test = delta_penalty
                                        # test = delta

                                        # if (tabuTenure[current_client] <= iter_ or delta + f < X_best):
                                        
                                        if(delta < best_delta):
                                            best_delta = delta
                                            best_idx1_client = idx1_c
                                            best_idx1_voiture = idx1_v
                                            best_idx2_client = idx2_c
                                            best_idx2_voiture = idx2_v
                                            
                                            best_is_swap = 0
                                    
                                        

                                            
                                        #new_client_before_after_swap != current_client and 
                                        #new_client_before_after_swap != client_before and 
                                        #new_client_before_after_swap != client_after and
                                        #new_client_before != client_before and
                                        #and  new_client_before != current_client
                                        # and new_client_after != current_client
                                        # and new_client_after != client_after
                                        # and new_client_after != client_before
                                        
                                        #Eval swap move
                                        if (new_client_before != client_after    and  new_client_before != 0  ):
                                            

                                            new_client_before_after_swap = current_solution_local[idx2_c - 1, idx2_v]
                                            new_client_after = current_solution_local[idx2_c + 1, idx2_v]
                                        
                                        
                                            client_swap = new_client_before
                                            
                                            #if (tabuTenure[client_swap - 1] <= iter_):
                                                
 
                                            impact_distance_client_swap_leaving = distance_matrix_gpu[new_client_before_after_swap, current_client] - distance_matrix_gpu[
                                    new_client_before_after_swap, client_swap] - distance_matrix_gpu[client_swap, current_client]
                                                                
                                                                
                                            impact_distance_client_swap_arrival = distance_matrix_gpu[client_before, client_swap] +   distance_matrix_gpu[client_swap, client_after] - distance_matrix_gpu[client_before, client_after]
                                                                
                                                                
                                                                
                                    
                                            
                                            
                                            
                                            delta = impact_client_leaving + impact_distance_client_arrival + impact_distance_client_swap_leaving + impact_distance_client_swap_arrival
                                            


                                            if (idx1_v != idx2_v):
                                                


                                                current_demande_local_tmp = current_demande_local[idx2_v] + client_demands_global_mem[current_client] 
                                                    

                                                if vehicle_capacity_global_mem[idx2_v] > current_demande_local_tmp - client_demands_global_mem[client_swap]:
                                                    demand_before_leaving = vehicle_capacity_global_mem[idx2_v]
                                                else:
                                                    demand_before_leaving = current_demande_local_tmp - client_demands_global_mem[ client_swap]    
                                                    
                                                    
                                                                
                                                if current_demande_local_tmp > demand_before_leaving:
                                                    impact_capacite_client_swap_leaving = current_demande_local_tmp - demand_before_leaving
                                                else:
                                                    impact_capacite_client_swap_leaving = 0   
                

                                                current_demande_local_tmp2 = current_demande_local[idx1_v] - client_demands_global_mem[int(current_client)] 


                                                if vehicle_capacity_global_mem[idx1_v] > current_demande_local_tmp2:
                                                    demand_after_arrival = vehicle_capacity_global_mem[idx1_v]
                                                else:
                                                    demand_after_arrival = current_demande_local_tmp2

                                                if current_demande_local_tmp2 + client_demands_global_mem[client_swap] > demand_after_arrival:
                                                    
                                                    impact_capacite_client_swap_arrival = current_demande_local_tmp2 + \
                                                                                    client_demands_global_mem[
                                                                                        client_swap] - demand_after_arrival
                                                else:
                                                    impact_capacite_client_swap_arrival = 0     
                                                
                                                
                                                delta_capacity = impact_capacite_client_arrival - impact_capacite_client_leaving + impact_capacite_client_swap_arrival - impact_capacite_client_swap_leaving 
                                                
                                                
                                                delta_penalty = lambda_ * delta_capacity
                                            
                                            
                                                delta +=  delta_penalty


        
                                            
                                            if(delta < best_delta):
                                                best_delta = delta
                                                best_idx1_client = idx1_c
                                                best_idx1_voiture = idx1_v
                                                best_idx2_client = idx2_c
                                                best_idx2_voiture = idx2_v
                                                
                                                best_is_swap = 1                                        
                                            


                f += best_delta
                # test = f

                # Mise à jour de la current solution
                best_client = current_solution_local[best_idx1_client, best_idx1_voiture]
                
                best_client_swap = current_solution_local[best_idx2_client, best_idx2_voiture]

                best_idx1_voiture_swap = best_idx2_voiture
                best_idx2_voiture_swap = best_idx1_voiture
                
                
                if best_idx1_voiture != best_idx2_voiture:
                    
                    best_idx1_client_swap = best_idx2_client
                    best_idx2_client_swap = best_idx1_client - 1
                    
                else:
                    
                    
                    if(best_idx2_client > best_idx1_client):
                        
                        best_idx1_client_swap = best_idx2_client - 1
                        best_idx2_client_swap = best_idx1_client - 1
                        
                    else:
                        
                        best_idx1_client_swap = best_idx2_client
                        best_idx2_client_swap = best_idx1_client
                        
                        
                
                                
                if best_client != 0 and best_client != -1:
                    
                    # Mise à jour de la route 1
                    for idx1_c in range(best_idx1_client, size_route_global_mem[d, best_idx1_voiture]):
                        current_solution_local[idx1_c, best_idx1_voiture] = current_solution_local[
                            idx1_c + 1, best_idx1_voiture]

                    if best_idx1_voiture == best_idx2_voiture and best_idx2_client > best_idx1_client:
                        best_idx2_client = best_idx2_client - 1

                    # Mise à jour de la route 2
                    for i in range(0, size_route_global_mem[d, best_idx2_voiture] - best_idx2_client):
                        idx2_c = size_route_global_mem[d, best_idx2_voiture] - i
                        current_solution_local[idx2_c + 1, best_idx2_voiture] = current_solution_local[
                            idx2_c, best_idx2_voiture]

                    current_solution_local[best_idx2_client + 1, best_idx2_voiture] = best_client

                    list_tabu = int(alpha * nb_clients + int((10 * xoroshiro128p_uniform_float32(rng_states, d)) + 1))

                    tabuTenure[best_client - 1] = iter_ + list_tabu

                    # Mise à jour de la size route
                    size_route_global_mem[d, best_idx1_voiture] -= 1
                    size_route_global_mem[d, best_idx2_voiture] += 1

                    # Mise à jour de la capacite
                    current_demande_local[best_idx1_voiture] -= client_demands_global_mem[best_client]
                    current_demande_local[best_idx2_voiture] += client_demands_global_mem[best_client]



                    #Mise à jour pour le deuxième client en cas de swap
                    if(best_is_swap == 1 and best_client_swap != 0 and best_client_swap != -1):

                
                        #Mise à jour de la route 1
                        for idx1_c in range(best_idx1_client_swap, size_route_global_mem[d, best_idx1_voiture_swap]):
                            current_solution_local[idx1_c, best_idx1_voiture_swap] = current_solution_local[
                                idx1_c + 1, best_idx1_voiture_swap]

                        if best_idx1_voiture_swap == best_idx2_voiture_swap and best_idx2_client_swap > best_idx1_client_swap:
                            best_idx2_client_swap = best_idx2_client_swap - 1

                        #Mise à jour de la route 2
                        for i in range(0, size_route_global_mem[d, best_idx2_voiture_swap] - best_idx2_client_swap):
                            idx2_c = size_route_global_mem[d, best_idx2_voiture_swap] - i
                            current_solution_local[idx2_c + 1, best_idx2_voiture_swap] = current_solution_local[
                                idx2_c, best_idx2_voiture_swap]

                        current_solution_local[best_idx2_client_swap + 1, best_idx2_voiture_swap] = best_client_swap

                        list_tabu = int(alpha * nb_clients + int((10 * xoroshiro128p_uniform_float32(rng_states, d)) + 1))

                        tabuTenure[best_client_swap - 1] = iter_ + list_tabu

                        #Mise à jour de la size route
                        size_route_global_mem[d, best_idx1_voiture_swap] -= 1
                        size_route_global_mem[d, best_idx2_voiture_swap] += 1

                        #Mise à jour de la capacite
                        current_demande_local[best_idx1_voiture_swap] -= client_demands_global_mem[best_client_swap]
                        current_demande_local[best_idx2_voiture_swap] += client_demands_global_mem[best_client_swap]





                # Mise à jour de la fitness et des structures en fonction du meilleur mouvement.
                if f < f_best:

                    f_best = f
                    # Mise à jour de la meilleure solution (best_solution)

                    for idx1_v in range(nb_voitures):
                        for idx1_c in range(nb_clients + 2):
                            best_iteration_solution_local[idx1_c, idx1_v] = current_solution_local[idx1_c, idx1_v]
                    #######
                    for idx1_v in range(nb_voitures):
                        best_iteration_demande_local[idx1_v] = current_demande_local[idx1_v]

                    islegal = True
                    for idx1_v in range(nb_voitures):
                        if (current_demande_local[idx1_v] > vehicle_capacity_global_mem[idx1_v]):
                            islegal = False

                    if (islegal and f < f_best_legal):
                        f_best_legal = f

                        for idx1_v in range(nb_voitures):
                            for idx1_c in range(nb_clients + 2):
                                current_solution_global_mem[d, idx1_c, idx1_v] = current_solution_local[idx1_c, idx1_v]

                # test = f_best

            # Ajustement de lambda_penalty en fonction de la demande finale

            islegal = True
            for idx1_v in range(nb_voitures):
                if best_iteration_demande_local[idx1_v] > vehicle_capacity_global_mem[idx1_v]:
                    islegal = False

            if islegal:
                
                lambda_ = lambda_//2
                if(lambda_ < 1):
                   lambda_ = 1
                
            else:
                lambda_ = lambda_ * 2

            for idx1_v in range(nb_voitures):
                for idx1_c in range(nb_clients + 2):
                    current_solution_local[idx1_c, idx1_v] = best_iteration_solution_local[idx1_c, idx1_v]

            for idx1_v in range(nb_voitures):
                current_demande_local[idx1_v] = best_iteration_demande_local[idx1_v]

            for idx1_v in range(nb_voitures):
                size_route_global_mem[d, idx1_v] = 0

            for idx1_v in range(nb_voitures):
                for idx1_c in range(nb_clients + 2):
                    if current_solution_local[idx1_c, idx1_v] != -1:
                        size_route_global_mem[d, idx1_v] += 1

        vector_f_global_mem[d] = f_best_legal


@cuda.jit
def tabu_CVRP_lambda_swap_NN_v2(rng_states, D, max_iter, distance_matrix_gpu, current_solution_global_mem,
                      demand_route_global_mem, vehicle_capacity_global_mem, client_demands_global_mem,
                      size_route_global_mem,  vector_f_global_mem, closest_clients_gpu_memory, lambda_, alpha, 
                      nb_iteration):
    d = cuda.grid(1)

    if d < D:

        tabuTenure = nb.cuda.local.array((nb_clients), dtype=nb.int16)


        current_solution_local = nb.cuda.local.array((size, nb_voitures), dtype=nb.int16)
        for idx1_v in range(nb_voitures):
            for idx1_c in range(nb_clients + 2):
                current_solution_local[idx1_c, idx1_v] = current_solution_global_mem[d, idx1_c, idx1_v]

        best_iteration_solution_local = nb.cuda.local.array((size, nb_voitures), dtype=nb.int16)

        for idx1_v in range(nb_voitures):
            for idx1_c in range(nb_clients + 2):
                best_iteration_solution_local[idx1_c, idx1_v] = current_solution_local[idx1_c, idx1_v]

        current_demande_local = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        for idx1_v in range(nb_voitures):
            current_demande_local[idx1_v] = demand_route_global_mem[d, idx1_v]

        best_iteration_demande_local = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        for idx1_v in range(nb_voitures):
            best_iteration_demande_local[idx1_v] = current_demande_local[idx1_v]

        f_best_legal = 99999



        position_clients = nb.cuda.local.array((nb_clients,2), dtype=nb.int16)
        

                    
                    


        ##########

        for iteration_global in range(2):

            if(iteration_global == 0):
                max_iter = 5000

            elif(iteration_global == 1):
                                
                lambda_ = 100
                
            
                max_iter = 5000
                
                
                
            for x in range(nb_clients):
                tabuTenure[x] = 0

            for idx1_v in range(nb_voitures):
                for idx1_c in range(nb_clients + 2):
                    
                    if(current_solution_local[idx1_c, idx1_v] != -1 and current_solution_local[idx1_c, idx1_v] != 0):
                        
                        client = current_solution_local[idx1_c, idx1_v] 
                        
                        position_clients[client-1,0] = idx1_c
                        position_clients[client-1,1] = idx1_v
                    
                    
            score_distance = 0

            total_penalty = 0

            for idx1_v in range(nb_voitures):

                route_demand = 0

                for idx1_c in range(int(size_route_global_mem[d, idx1_v] - 1)):
                    client = current_solution_local[idx1_c, idx1_v]
                    next_client = current_solution_local[idx1_c + 1, idx1_v]

                    score_distance += distance_matrix_gpu[client, next_client]
                    route_demand += client_demands_global_mem[client]

                if (route_demand > vehicle_capacity_global_mem[idx1_v]):
                    capacity_violation = route_demand - vehicle_capacity_global_mem[idx1_v]
                else:
                    capacity_violation = 0

                total_penalty += lambda_ * capacity_violation

            f = score_distance + total_penalty

            f_best = f

            ####### Incremental Moves

            for iter_ in range(max_iter):

                best_delta = 99999
                best_idx1_client = -1
                best_idx1_voiture = -1
                best_idx2_client = -1
                best_idx2_voiture = -1
                best_is_swap = 0

                #for idx1_v in range(nb_voitures):
                    #for idx1_c in range(1, int(size_route_global_mem[d, idx1_v] - 1)):
                    
                for current_client in range(1, nb_clients + 1):
                    
                    #current_client = current_solution_local[idx1_c, idx1_v]
                
                    idx1_c = position_clients[current_client-1,0] 
                    idx1_v = position_clients[current_client-1,1] 
                    
                    
                    if (tabuTenure[current_client - 1] <= iter_):
                        
                        ### A améliorer, peut être précalculé peut être
                        client_before = current_solution_local[idx1_c - 1, idx1_v]
                        client_after = current_solution_local[idx1_c + 1, idx1_v]
                        
                        impact_client_leaving = distance_matrix_gpu[client_before, client_after] - distance_matrix_gpu[
                            current_client, client_before] - distance_matrix_gpu[current_client, client_after]


                        if vehicle_capacity_global_mem[idx1_v] > current_demande_local[idx1_v] - \
                                client_demands_global_mem[current_client]:
                            demand_before_leaving = vehicle_capacity_global_mem[idx1_v]
                        else:
                            demand_before_leaving = current_demande_local[idx1_v] - client_demands_global_mem[
                                current_client]
                            
                            
                        if current_demande_local[idx1_v] > demand_before_leaving:
                            impact_capacite_client_leaving = current_demande_local[idx1_v] - demand_before_leaving
                        else:
                            impact_capacite_client_leaving = 0


                        #for idx2_v in range(nb_voitures):
                            #for idx2_c in range(int(size_route_global_mem[d, idx2_v] - 1)):
                        
                        
                        
                        

                        for idx2_v in range(nb_voitures):      

                            idx2_c = 0
                            new_client_before = current_solution_local[idx2_c, idx2_v]
                            
                            #new_client_before = closest_clients_gpu_memory[d,i]
                            
                            #idx2_c = position_clients[new_client_before-1,0] 
                            #idx2_v = position_clients[new_client_before-1,1]                             


                            ### Eval relocate move
                            if new_client_before != current_client and new_client_before != client_before:

                                new_client_after = current_solution_local[idx2_c + 1, idx2_v]
                                impact_distance_client_arrival = distance_matrix_gpu[new_client_before, current_client] + \
                                                        distance_matrix_gpu[current_client, new_client_after] - \
                                                        distance_matrix_gpu[new_client_before, new_client_after]



                                delta = impact_client_leaving + impact_distance_client_arrival
                                # test = delta

                                # Ajout de la pénalité liée à la capacité dans delta
                                if (idx1_v != idx2_v):
                                    
                                    # Impact de la capacité en arrivant au nouveau client
                                    if vehicle_capacity_global_mem[idx2_v] > current_demande_local[idx2_v]:
                                        demand_after_arrival = vehicle_capacity_global_mem[idx2_v]
                                    else:
                                        demand_after_arrival = current_demande_local[idx2_v]

                                    if current_demande_local[idx2_v] + client_demands_global_mem[current_client] > demand_after_arrival:
                                        
                                        impact_capacite_client_arrival = current_demande_local[idx2_v] + \
                                                                        client_demands_global_mem[
                                                                            current_client] - demand_after_arrival
                                    else:
                                        impact_capacite_client_arrival = 0
                                    
                                    delta_capacity = impact_capacite_client_arrival - impact_capacite_client_leaving
                                    delta_penalty = lambda_ * delta_capacity
                                    delta += delta_penalty
                                    # test = delta_penalty
                                # test = delta

                                # if (tabuTenure[current_client] <= iter_ or delta + f < X_best):
                                
                                if(delta < best_delta):
                                    best_delta = delta
                                    best_idx1_client = idx1_c
                                    best_idx1_voiture = idx1_v
                                    best_idx2_client = idx2_c
                                    best_idx2_voiture = idx2_v
                                    
                                    best_is_swap = 0
                                    
                                    
                        ############
                        #for i in range(nb_nearest_neighbor_tabu):        
                        
                        #for idx2_v in range(nb_voitures):
                            #for idx2_c in range(1, int(size_route_global_mem[d, idx2_v]) - 1):
                        for i in range(nb_nearest_neighbor_tabu):       
                        #for new_client_before in range(1, nb_clients + 1):
                            
                            #new_client_before = current_solution_local[idx2_c, idx2_v]
                            new_client_before = closest_clients_gpu_memory[current_client-1,i] 
                            
                            idx2_c = position_clients[new_client_before-1,0] 
                            idx2_v = position_clients[new_client_before-1,1]                             


                            ### Eval relocate move
                            if new_client_before != current_client and new_client_before != client_before:

                                new_client_after = current_solution_local[idx2_c + 1, idx2_v]
                                impact_distance_client_arrival = distance_matrix_gpu[new_client_before, current_client] + \
                                                        distance_matrix_gpu[current_client, new_client_after] - \
                                                        distance_matrix_gpu[new_client_before, new_client_after]



                                delta = impact_client_leaving + impact_distance_client_arrival
                                # test = delta

                                # Ajout de la pénalité liée à la capacité dans delta
                                if (idx1_v != idx2_v):
                                    
                                    # Impact de la capacité en arrivant au nouveau client
                                    if vehicle_capacity_global_mem[idx2_v] > current_demande_local[idx2_v]:
                                        demand_after_arrival = vehicle_capacity_global_mem[idx2_v]
                                    else:
                                        demand_after_arrival = current_demande_local[idx2_v]

                                    if current_demande_local[idx2_v] + client_demands_global_mem[current_client] > demand_after_arrival:
                                        
                                        impact_capacite_client_arrival = current_demande_local[idx2_v] + \
                                                                        client_demands_global_mem[
                                                                            current_client] - demand_after_arrival
                                    else:
                                        impact_capacite_client_arrival = 0
                                    
                                    delta_capacity = impact_capacite_client_arrival - impact_capacite_client_leaving
                                    delta_penalty = lambda_ * delta_capacity
                                    delta += delta_penalty
                                    # test = delta_penalty
                                # test = delta

                                # if (tabuTenure[current_client] <= iter_ or delta + f < X_best):
                                
                                if(delta < best_delta):
                                    best_delta = delta
                                    best_idx1_client = idx1_c
                                    best_idx1_voiture = idx1_v
                                    best_idx2_client = idx2_c
                                    best_idx2_voiture = idx2_v
                                    
                                    best_is_swap = 0
                            
                                
                                #Eval swap move
                                if (new_client_before != client_after    and  new_client_before != 0  ):
                                    
                                    
                                    client_swap = new_client_before
                                    
                                    #if (tabuTenure[client_swap - 1] <= iter_):
                                        
                                    new_client_before_after_swap = current_solution_local[idx2_c - 1, idx2_v]
                                    new_client_after = current_solution_local[idx2_c + 1, idx2_v]
                                    




                                    impact_distance_client_swap_leaving = distance_matrix_gpu[new_client_before_after_swap, current_client] - distance_matrix_gpu[
                            new_client_before_after_swap, client_swap] - distance_matrix_gpu[client_swap, current_client]
                                                        
                                                        
                                    impact_distance_client_swap_arrival = distance_matrix_gpu[client_before, client_swap] +   distance_matrix_gpu[client_swap, client_after] - distance_matrix_gpu[client_before, client_after]
                                                        
                                                        
                                                        
                            
                                    
                                    
                                    
                                    delta = impact_client_leaving + impact_distance_client_arrival + impact_distance_client_swap_leaving + impact_distance_client_swap_arrival
                                    


                                    if (idx1_v != idx2_v):
                                        


                                        current_demande_local_tmp = current_demande_local[idx2_v] + client_demands_global_mem[current_client] 
                                            

                                        if vehicle_capacity_global_mem[idx2_v] > current_demande_local_tmp - client_demands_global_mem[client_swap]:
                                            demand_before_leaving = vehicle_capacity_global_mem[idx2_v]
                                        else:
                                            demand_before_leaving = current_demande_local_tmp - client_demands_global_mem[ client_swap]    
                                            
                                            
                                                        
                                        if current_demande_local_tmp > demand_before_leaving:
                                            impact_capacite_client_swap_leaving = current_demande_local_tmp - demand_before_leaving
                                        else:
                                            impact_capacite_client_swap_leaving = 0   
        

                                        current_demande_local_tmp2 = current_demande_local[idx1_v] - client_demands_global_mem[int(current_client)] 


                                        if vehicle_capacity_global_mem[idx1_v] > current_demande_local_tmp2:
                                            demand_after_arrival = vehicle_capacity_global_mem[idx1_v]
                                        else:
                                            demand_after_arrival = current_demande_local_tmp2

                                        if current_demande_local_tmp2 + client_demands_global_mem[client_swap] > demand_after_arrival:
                                            
                                            impact_capacite_client_swap_arrival = current_demande_local_tmp2 + \
                                                                            client_demands_global_mem[
                                                                                client_swap] - demand_after_arrival
                                        else:
                                            impact_capacite_client_swap_arrival = 0     
                                        
                                        
                                        delta_capacity = impact_capacite_client_arrival - impact_capacite_client_leaving + impact_capacite_client_swap_arrival - impact_capacite_client_swap_leaving 
                                        
                                        
                                        delta_penalty = lambda_ * delta_capacity
                                        delta +=  delta_penalty



                                    
                                    if(delta < best_delta):
                                        best_delta = delta
                                        best_idx1_client = idx1_c
                                        best_idx1_voiture = idx1_v
                                        best_idx2_client = idx2_c
                                        best_idx2_voiture = idx2_v
                                        
                                        best_is_swap = 1                                        
                                        



                f += best_delta
                # test = f

                # Mise à jour de la current solution
                best_client = current_solution_local[best_idx1_client, best_idx1_voiture]
                
                best_client_swap = current_solution_local[best_idx2_client, best_idx2_voiture]

                best_idx1_voiture_swap = best_idx2_voiture
                best_idx2_voiture_swap = best_idx1_voiture
                
                
                if best_idx1_voiture != best_idx2_voiture:
                    
                    best_idx1_client_swap = best_idx2_client
                    best_idx2_client_swap = best_idx1_client - 1
                    
                else:
                    
                    
                    if(best_idx2_client > best_idx1_client):
                        
                        best_idx1_client_swap = best_idx2_client - 1
                        best_idx2_client_swap = best_idx1_client - 1
                        
                    else:
                        
                        best_idx1_client_swap = best_idx2_client
                        best_idx2_client_swap = best_idx1_client
                        
                        
                
                                
                if best_client != 0 and best_client != -1:
                    
                    # Mise à jour de la route 1
                    for idx1_c in range(best_idx1_client, size_route_global_mem[d, best_idx1_voiture]):
                        
                        client = current_solution_local[idx1_c + 1, best_idx1_voiture] 
                        
                        if(client != 0 and client != -1):
                            position_clients[client-1,0] = idx1_c
                            position_clients[client-1,1] = best_idx1_voiture
                        
                        current_solution_local[idx1_c, best_idx1_voiture] = current_solution_local[
                            idx1_c + 1, best_idx1_voiture]
                        
                        
                        

                    if best_idx1_voiture == best_idx2_voiture and best_idx2_client > best_idx1_client:
                        best_idx2_client = best_idx2_client - 1

                    # Mise à jour de la route 2
                    for i in range(0, size_route_global_mem[d, best_idx2_voiture] - best_idx2_client):
                        idx2_c = size_route_global_mem[d, best_idx2_voiture] - i
                        
                        client = current_solution_local[idx2_c, best_idx2_voiture] 
                        if(client != 0 and client != -1):
                            position_clients[client-1,0] = idx2_c + 1
                            position_clients[client-1,1] = best_idx2_voiture
                        
                        current_solution_local[idx2_c + 1, best_idx2_voiture] = current_solution_local[idx2_c, best_idx2_voiture]

                    
                    position_clients[best_client-1,0] = best_idx2_client + 1
                    position_clients[best_client-1,1] = best_idx2_voiture
                        
                    current_solution_local[best_idx2_client + 1, best_idx2_voiture] = best_client

                    list_tabu = int(alpha * nb_clients + int((10 * xoroshiro128p_uniform_float32(rng_states, d)) + 1))

                    tabuTenure[best_client - 1] = iter_ + list_tabu

                    # Mise à jour de la size route
                    size_route_global_mem[d, best_idx1_voiture] -= 1
                    size_route_global_mem[d, best_idx2_voiture] += 1

                    # Mise à jour de la capacite
                    current_demande_local[best_idx1_voiture] -= client_demands_global_mem[best_client]
                    current_demande_local[best_idx2_voiture] += client_demands_global_mem[best_client]



                    #Mise à jour pour le deuxième client en cas de swap
                    if(best_is_swap == 1 and best_client_swap != 0 and best_client_swap != -1):

                
                        #Mise à jour de la route 1
                        for idx1_c in range(best_idx1_client_swap, size_route_global_mem[d, best_idx1_voiture_swap]):
                            
                            client = current_solution_local[idx1_c + 1, best_idx1_voiture_swap] 
                            
                            if(client != 0 and client != -1):
                                position_clients[client-1,0] = idx1_c
                                position_clients[client-1,1] = best_idx1_voiture_swap
                        
                            current_solution_local[idx1_c, best_idx1_voiture_swap] = current_solution_local[
                                idx1_c + 1, best_idx1_voiture_swap]

                        if best_idx1_voiture_swap == best_idx2_voiture_swap and best_idx2_client_swap > best_idx1_client_swap:
                            best_idx2_client_swap = best_idx2_client_swap - 1

                        #Mise à jour de la route 2
                        for i in range(0, size_route_global_mem[d, best_idx2_voiture_swap] - best_idx2_client_swap):
                            idx2_c = size_route_global_mem[d, best_idx2_voiture_swap] - i
                            
                            client = current_solution_local[idx2_c, best_idx2_voiture_swap]
                            
                            if(client != 0 and client != -1):
                                position_clients[client-1,0] = idx2_c + 1
                                position_clients[client-1,1] = best_idx2_voiture_swap
                        
                            current_solution_local[idx2_c + 1, best_idx2_voiture_swap] = current_solution_local[
                                idx2_c, best_idx2_voiture_swap]

                        
                        position_clients[best_client_swap-1,0] = best_idx2_client_swap + 1
                        position_clients[best_client_swap-1,1] = best_idx2_voiture_swap
                    
                        current_solution_local[best_idx2_client_swap + 1, best_idx2_voiture_swap] = best_client_swap

                        list_tabu = int(alpha * nb_clients + int((10 * xoroshiro128p_uniform_float32(rng_states, d)) + 1))

                        tabuTenure[best_client_swap - 1] = iter_ + list_tabu

                        #Mise à jour de la size route
                        size_route_global_mem[d, best_idx1_voiture_swap] -= 1
                        size_route_global_mem[d, best_idx2_voiture_swap] += 1

                        #Mise à jour de la capacite
                        current_demande_local[best_idx1_voiture_swap] -= client_demands_global_mem[best_client_swap]
                        current_demande_local[best_idx2_voiture_swap] += client_demands_global_mem[best_client_swap]





                # Mise à jour de la fitness et des structures en fonction du meilleur mouvement.
                if f < f_best:

                    f_best = f
                    # Mise à jour de la meilleure solution (best_solution)

                    for idx1_v in range(nb_voitures):
                        for idx1_c in range(nb_clients + 2):
                            best_iteration_solution_local[idx1_c, idx1_v] = current_solution_local[idx1_c, idx1_v]
                    #######
                    for idx1_v in range(nb_voitures):
                        best_iteration_demande_local[idx1_v] = current_demande_local[idx1_v]



                # test = f_best

            # Ajustement de lambda_penalty en fonction de la demande finale

            #islegal = True
            #for idx1_v in range(nb_voitures):
                #if best_iteration_demande_local[idx1_v] > vehicle_capacity_global_mem[idx1_v]:
                    #islegal = False

            #if islegal:
                
                #lambda_ = lambda_ * 0.5
                #if(lambda_ < 1):
                   #lambda_ = 1
                
            #else:
                #lambda_ = lambda_ * 2.0







            for idx1_v in range(nb_voitures):
                for idx1_c in range(nb_clients + 2):
                    current_solution_local[idx1_c, idx1_v] = best_iteration_solution_local[idx1_c, idx1_v]

            for idx1_v in range(nb_voitures):
                current_demande_local[idx1_v] = best_iteration_demande_local[idx1_v]

            for idx1_v in range(nb_voitures):
                size_route_global_mem[d, idx1_v] = 0

            for idx1_v in range(nb_voitures):
                for idx1_c in range(nb_clients + 2):
                    if current_solution_local[idx1_c, idx1_v] != -1:
                        size_route_global_mem[d, idx1_v] += 1


        islegal = True
        for idx1_v in range(nb_voitures):
            if (best_iteration_demande_local[idx1_v] > vehicle_capacity_global_mem[idx1_v]):
                islegal = False

        if (islegal and f < f_best_legal):
            f_best_legal = f

            for idx1_v in range(nb_voitures):
                for idx1_c in range(nb_clients + 2):
                    current_solution_global_mem[d, idx1_c, idx1_v] = best_iteration_solution_local[idx1_c, idx1_v]
                        
                        
        vector_f_global_mem[d] = f_best_legal
        
        

#@cuda.jit
#def tabu_CVRP_lambda_swap_v2(rng_states, D, max_iter, distance_matrix_gpu, current_solution_global_mem,
                      #demand_route_global_mem, vehicle_capacity_global_mem, client_demands_global_mem,
                      #size_route_global_mem,  vector_f_global_mem, lambda_, alpha, 
                      #nb_iteration):
    #d = cuda.grid(1)

    #if d < D:

        #tabuTenure = nb.cuda.local.array((nb_clients), dtype=nb.int16)


        #current_solution_local = nb.cuda.local.array((size, nb_voitures), dtype=nb.int16)
        #for idx1_v in range(nb_voitures):
            #for idx1_c in range(nb_clients + 2):
                #current_solution_local[idx1_c, idx1_v] = current_solution_global_mem[d, idx1_c, idx1_v]

        #best_iteration_solution_local = nb.cuda.local.array((size, nb_voitures), dtype=nb.int16)

        #for idx1_v in range(nb_voitures):
            #for idx1_c in range(nb_clients + 2):
                #best_iteration_solution_local[idx1_c, idx1_v] = current_solution_local[idx1_c, idx1_v]

        #current_demande_local = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        #for idx1_v in range(nb_voitures):
            #current_demande_local[idx1_v] = demand_route_global_mem[d, idx1_v]

        #best_iteration_demande_local = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        #for idx1_v in range(nb_voitures):
            #best_iteration_demande_local[idx1_v] = current_demande_local[idx1_v]

        #f_best_legal = 99999

        

        #for _ in range(nb_iteration):

            #for x in range(nb_clients):
                #tabuTenure[x] = 0

            #score_distance = 0

            #total_penalty = 0

            #for idx1_v in range(nb_voitures):

                #route_demand = 0

                #for idx1_c in range(int(size_route_global_mem[d, idx1_v] - 1)):
                    #client = current_solution_local[idx1_c, idx1_v]
                    #next_client = current_solution_local[idx1_c + 1, idx1_v]

                    #score_distance += distance_matrix_gpu[client, next_client]
                    #route_demand += client_demands_global_mem[client]

                #if (route_demand > vehicle_capacity_global_mem[idx1_v]):
                    #capacity_violation = route_demand - vehicle_capacity_global_mem[idx1_v]
                #else:
                    #capacity_violation = 0

                #total_penalty += lambda_ * capacity_violation

            #f = score_distance + total_penalty

            #f_best = f

            #Incremental Moves

            #for iter_ in range(max_iter):

                #best_delta = 99999
                #best_idx1_client = -1
                #best_idx1_voiture = -1
                #best_idx2_client = -1
                #best_idx2_voiture = -1
                #best_is_swap = 0

                #for idx1_v in range(nb_voitures):
                    #for idx1_c in range(1, int(size_route_global_mem[d, idx1_v] - 1)):
                        #Leaving client x
                        #current_client = current_solution_local[idx1_c, idx1_v]
                        
                        #if (tabuTenure[current_client - 1] <= iter_):
                            
                            #client_before = current_solution_local[idx1_c - 1, idx1_v]
                            #client_after = current_solution_local[idx1_c + 1, idx1_v]
                            #impact_client_leaving = distance_matrix_gpu[client_before, client_after] - distance_matrix_gpu[
                                #current_client, client_before] - distance_matrix_gpu[current_client, client_after]

                            #Impact de la capacité en quittant le client
                            #if vehicle_capacity_global_mem[idx1_v] > current_demande_local[idx1_v] - \
                                    #client_demands_global_mem[current_client]:
                                #demand_before_leaving = vehicle_capacity_global_mem[idx1_v]
                            #else:
                                #demand_before_leaving = current_demande_local[idx1_v] - client_demands_global_mem[
                                    #current_client]
                                
                                
                            #if current_demande_local[idx1_v] > demand_before_leaving:
                                #impact_capacite_client_leaving = current_demande_local[idx1_v] - demand_before_leaving
                            #else:
                                #impact_capacite_client_leaving = 0


                            #for idx2_v in range(nb_voitures):
                                #for idx2_c in range(int(size_route_global_mem[d, idx2_v] - 1)):
                                    #new position for client x
                                    #new_client_before = current_solution_local[idx2_c, idx2_v]


                                    #Eval relocate move
                                    #if new_client_before != current_client and new_client_before != client_before:

                                        #new_client_after = current_solution_local[idx2_c + 1, idx2_v]
                                        #impact_distance_client_arrival = distance_matrix_gpu[new_client_before, current_client] + \
                                                                #distance_matrix_gpu[current_client, new_client_after] - \
                                                                #distance_matrix_gpu[new_client_before, new_client_after]



                                        #delta = impact_client_leaving + impact_distance_client_arrival
                                        #test = delta

                                        #Ajout de la pénalité liée à la capacité dans delta
                                        #if (idx1_v != idx2_v):
                                            
                                            #Impact de la capacité en arrivant au nouveau client
                                            #if vehicle_capacity_global_mem[idx2_v] > current_demande_local[idx2_v]:
                                                #demand_after_arrival = vehicle_capacity_global_mem[idx2_v]
                                            #else:
                                                #demand_after_arrival = current_demande_local[idx2_v]

                                            #if current_demande_local[idx2_v] + client_demands_global_mem[current_client] > demand_after_arrival:
                                                
                                                #impact_capacite_client_arrival = current_demande_local[idx2_v] + \
                                                                                #client_demands_global_mem[
                                                                                    #current_client] - demand_after_arrival
                                            #else:
                                                #impact_capacite_client_arrival = 0
                                            
                                            #delta_capacity = impact_capacite_client_arrival - impact_capacite_client_leaving
                                            #delta_penalty = lambda_ * delta_capacity
                                            #delta += delta_penalty
                                            #test = delta_penalty
                                        #test = delta

                                        #if (tabuTenure[current_client] <= iter_ or delta + f < X_best):
                                        
                                        #if(delta < best_delta):
                                            #best_delta = delta
                                            #best_idx1_client = idx1_c
                                            #best_idx1_voiture = idx1_v
                                            #best_idx2_client = idx2_c
                                            #best_idx2_voiture = idx2_v
                                            
                                            #best_is_swap = 0
                                    
                                    
                                    #Eval swap move
                                    #if  new_client_before != current_client  and new_client_before != client_before and new_client_before != 0:
                                        
                                        
                                        #client_swap = new_client_before
                                        
                                        #if (tabuTenure[client_swap - 1] <= iter_):
                                            
                                        #new_client_before_after_swap = current_solution_local[idx2_c - 1, idx2_v]
                                        #new_client_after = current_solution_local[idx2_c + 1, idx2_v]
                                        




                                        #impact_distance_client_swap_leaving = distance_matrix_gpu[new_client_before_after_swap, current_client] - distance_matrix_gpu[
                                #new_client_before_after_swap, client_swap] - distance_matrix_gpu[client_swap, current_client]
                                                            
                                                            
                                        #impact_distance_client_swap_arrival = distance_matrix_gpu[client_before, client_swap] +   distance_matrix_gpu[client_swap, client_after] - distance_matrix_gpu[client_before, client_after]
                                                            
                                                            
                                                            
                                
                                        
                                        
                                        
                                        #delta_distance = impact_client_leaving + impact_distance_client_arrival + impact_distance_client_swap_leaving + impact_distance_client_swap_arrival
                                        


                                        #if (idx1_v != idx2_v):
                                            


                                            #current_demande_local_tmp = current_demande_local[idx2_v] + client_demands_global_mem[current_client] 
                                                

                                            #if vehicle_capacity_global_mem[idx2_v] > current_demande_local_tmp - client_demands_global_mem[client_swap]:
                                                #demand_before_leaving = vehicle_capacity_global_mem[idx2_v]
                                            #else:
                                                #demand_before_leaving = current_demande_local_tmp - client_demands_global_mem[ client_swap]    
                                                
                                                
                                                            
                                            #if current_demande_local_tmp > demand_before_leaving:
                                                #impact_capacite_client_swap_leaving = current_demande_local_tmp - demand_before_leaving
                                            #else:
                                                #impact_capacite_client_swap_leaving = 0   
            

                                            #current_demande_local_tmp2 = current_demande_local[idx1_v] - client_demands_global_mem[int(current_client)] 


                                            #if vehicle_capacity_global_mem[idx1_v] > current_demande_local_tmp2:
                                                #demand_after_arrival = vehicle_capacity_global_mem[idx1_v]
                                            #else:
                                                #demand_after_arrival = current_demande_local_tmp2

                                            #if current_demande_local_tmp2 + client_demands_global_mem[client_swap] > demand_after_arrival:
                                                
                                                #impact_capacite_client_swap_arrival = current_demande_local_tmp2 + \
                                                                                #client_demands_global_mem[
                                                                                    #client_swap] - demand_after_arrival
                                            #else:
                                                #impact_capacite_client_swap_arrival = 0     
                                            
                                            
                                            #delta_capacity = impact_capacite_client_arrival - impact_capacite_client_leaving + impact_capacite_client_swap_arrival - impact_capacite_client_swap_leaving 
                                            
                                            
                                            #delta_penalty = lambda_ * delta_capacity
                                            #delta = delta_distance + delta_penalty


    
                                        
                                            #if(delta < best_delta):
                                                #best_delta = delta
                                                #best_idx1_client = idx1_c
                                                #best_idx1_voiture = idx1_v
                                                #best_idx2_client = idx2_c
                                                #best_idx2_voiture = idx2_v
                                                
                                                #best_is_swap = 1                                        
                                            


                #f += best_delta
                #test = f

                #Mise à jour de la current solution
                #best_client = current_solution_local[best_idx1_client, best_idx1_voiture]
                
                #best_client_swap = current_solution_local[best_idx2_client, best_idx2_voiture]

                #best_idx1_voiture_swap = best_idx2_voiture
                #best_idx2_voiture_swap = best_idx1_voiture
                
                
                #if best_idx1_voiture != best_idx2_voiture:
                    
                    #best_idx1_client_swap = best_idx2_client
                    #best_idx2_client_swap = best_idx1_client - 1
                    
                #else:
                    
                    #if(best_idx2_client > best_idx1_client):
                        
                        #best_idx1_client_swap = best_idx2_client - 1
                        #best_idx2_client_swap = best_idx1_client - 1
                        
                    #else:
                        
                        #best_idx1_client_swap = best_idx2_client
                        #best_idx2_client_swap = best_idx1_client
                        
                        
                
                                
                #if best_client != 0 and best_client != -1:
                    
                    #Mise à jour de la route 1
                    #for idx1_c in range(best_idx1_client, size_route_global_mem[d, best_idx1_voiture]):
                        #current_solution_local[idx1_c, best_idx1_voiture] = current_solution_local[
                            #idx1_c + 1, best_idx1_voiture]

                    #if best_idx1_voiture == best_idx2_voiture and best_idx2_client > best_idx1_client:
                        #best_idx2_client = best_idx2_client - 1

                    #Mise à jour de la route 2
                    #for i in range(0, size_route_global_mem[d, best_idx2_voiture] - best_idx2_client):
                        #idx2_c = size_route_global_mem[d, best_idx2_voiture] - i
                        #current_solution_local[idx2_c + 1, best_idx2_voiture] = current_solution_local[
                            #idx2_c, best_idx2_voiture]

                    #current_solution_local[best_idx2_client + 1, best_idx2_voiture] = best_client

                    #list_tabu = int(alpha * nb_clients + int((10 * xoroshiro128p_uniform_float32(rng_states, d)) + 1))

                    #tabuTenure[best_client - 1] = iter_ + list_tabu

                    #Mise à jour de la size route
                    #size_route_global_mem[d, best_idx1_voiture] -= 1
                    #size_route_global_mem[d, best_idx2_voiture] += 1

                    #Mise à jour de la capacite
                    #current_demande_local[best_idx1_voiture] -= client_demands_global_mem[best_client]
                    #current_demande_local[best_idx2_voiture] += client_demands_global_mem[best_client]



                    #Mise à jour pour le deuxième client en cas de swap
                    #if(best_is_swap == 1 and best_client_swap != 0 and best_client_swap != -1):

                
                        #Mise à jour de la route 1
                        #for idx1_c in range(best_idx1_client_swap, size_route_global_mem[d, best_idx1_voiture_swap]):
                            #current_solution_local[idx1_c, best_idx1_voiture_swap] = current_solution_local[
                                #idx1_c + 1, best_idx1_voiture_swap]

                        #if best_idx1_voiture_swap == best_idx2_voiture_swap and best_idx2_client_swap > best_idx1_client_swap:
                            #best_idx2_client_swap = best_idx2_client_swap - 1

                        #Mise à jour de la route 2
                        #for i in range(0, size_route_global_mem[d, best_idx2_voiture_swap] - best_idx2_client_swap):
                            #idx2_c = size_route_global_mem[d, best_idx2_voiture_swap] - i
                            #current_solution_local[idx2_c + 1, best_idx2_voiture_swap] = current_solution_local[
                                #idx2_c, best_idx2_voiture_swap]

                        #current_solution_local[best_idx2_client_swap + 1, best_idx2_voiture_swap] = best_client_swap

                        #list_tabu = int(alpha * nb_clients + int((10 * xoroshiro128p_uniform_float32(rng_states, d)) + 1))

                        #tabuTenure[best_client_swap - 1] = iter_ + list_tabu

                        #Mise à jour de la size route
                        #size_route_global_mem[d, best_idx1_voiture_swap] -= 1
                        #size_route_global_mem[d, best_idx2_voiture_swap] += 1

                        #Mise à jour de la capacite
                        #current_demande_local[best_idx1_voiture_swap] -= client_demands_global_mem[best_client_swap]
                        #current_demande_local[best_idx2_voiture_swap] += client_demands_global_mem[best_client_swap]





                #Mise à jour de la fitness et des structures en fonction du meilleur mouvement.
                #if f < f_best:

                    #f_best = f
                    #Mise à jour de la meilleure solution (best_solution)

                    #for idx1_v in range(nb_voitures):
                        #for idx1_c in range(nb_clients + 2):
                            #best_iteration_solution_local[idx1_c, idx1_v] = current_solution_local[idx1_c, idx1_v]
                    
                    #for idx1_v in range(nb_voitures):
                        #best_iteration_demande_local[idx1_v] = current_demande_local[idx1_v]

                    #islegal = True
                    #for idx1_v in range(nb_voitures):
                        #if (current_demande_local[idx1_v] > vehicle_capacity_global_mem[idx1_v]):
                            #islegal = False

                    #if (islegal and f < f_best_legal):
                        #f_best_legal = f

                        #for idx1_v in range(nb_voitures):
                            #for idx1_c in range(nb_clients + 2):
                                #current_solution_global_mem[d, idx1_c, idx1_v] = current_solution_local[idx1_c, idx1_v]

                #test = f_best

            #Ajustement de lambda_penalty en fonction de la demande finale

            #islegal = True
            #for idx1_v in range(nb_voitures):
                #if best_iteration_demande_local[idx1_v] > vehicle_capacity_global_mem[idx1_v]:
                    #islegal = False

            #if islegal:
                
                #lambda_ = lambda_ - 1
                #if(lambda_ < 1):
                   #lambda_ = 1
                
            #else:
                #lambda_ = lambda_ + 1 

            #for idx1_v in range(nb_voitures):
                #for idx1_c in range(nb_clients + 2):
                    #current_solution_local[idx1_c, idx1_v] = best_iteration_solution_local[idx1_c, idx1_v]

            #for idx1_v in range(nb_voitures):
                #current_demande_local[idx1_v] = best_iteration_demande_local[idx1_v]

            #for idx1_v in range(nb_voitures):
                #size_route_global_mem[d, idx1_v] = 0

            #for idx1_v in range(nb_voitures):
                #for idx1_c in range(nb_clients + 2):
                    #if current_solution_local[idx1_c, idx1_v] != -1:
                        #size_route_global_mem[d, idx1_v] += 1

        #vector_f_global_mem[d] = f_best_legal


@cuda.jit
def tabu_CVRP_lambda_swap_NN(rng_states, D, max_iter, distance_matrix_gpu, current_solution_global_mem,
                      demand_route_global_mem, vehicle_capacity_global_mem, client_demands_global_mem,
                      size_route_global_mem,  vector_f_global_mem, closest_clients_gpu_memory, lambda_, factor_lambda, alpha, 
                      nb_iteration):
    d = cuda.grid(1)

    if d < D:

        # Initialisation des structures de données locales
        tabuTenure = nb.cuda.local.array((nb_clients), dtype=nb.int16)
        current_solution_local = nb.cuda.local.array((size, nb_voitures), dtype=nb.int16)
        best_iteration_solution_local = nb.cuda.local.array((size, nb_voitures), dtype=nb.int16)        
        current_demande_local = nb.cuda.local.array((nb_voitures), dtype=nb.int16)        
        best_iteration_demande_local = nb.cuda.local.array((nb_voitures), dtype=nb.int16)        
        position_clients = nb.cuda.local.array((nb_clients,2), dtype=nb.int16)  
        
        # Copie de la solution courante dans les structures locales        
        for idx1_v in range(nb_voitures):
            for idx1_c in range(nb_clients + 2):
                current_solution_local[idx1_c, idx1_v] = current_solution_global_mem[d, idx1_c, idx1_v]
                
        for idx1_v in range(nb_voitures):
            for idx1_c in range(nb_clients + 2):                
                best_iteration_solution_local[idx1_c, idx1_v] = current_solution_local[idx1_c, idx1_v]

        # Copie de la demande courante dans les structures locales
        for idx1_v in range(nb_voitures):
            current_demande_local[idx1_v] = demand_route_global_mem[d, idx1_v]
            
        for idx1_v in range(nb_voitures):           
            best_iteration_demande_local[idx1_v] = current_demande_local[idx1_v]

        # Initialisation de la meilleure fitness légale
        f_best_legal = infinit

        ##########

        for global_iter in range(nb_iteration):

            # Réinitialisation de la liste Tabu
            for x in range(nb_clients):
                tabuTenure[x] = 0

            # Mise à jour des positions des clients dans la solution courante
            for idx1_v in range(nb_voitures):
                for idx1_c in range(nb_clients + 2):
                    if(current_solution_local[idx1_c, idx1_v] != -1 and current_solution_local[idx1_c, idx1_v] != 0):
                        client = current_solution_local[idx1_c, idx1_v] 
                        position_clients[client-1, 0] = idx1_c
                        position_clients[client-1, 1] = idx1_v
                    
            # Initialisation du score de distance et de la pénalité totale                   
            score_distance = 0
            total_penalty = 0

            # Calcul du score de distance et de la pénalité totale
            for idx1_v in range(nb_voitures):
                route_demand = 0
                for idx1_c in range(int(size_route_global_mem[d, idx1_v] - 1)):
                    client = current_solution_local[idx1_c, idx1_v]
                    next_client = current_solution_local[idx1_c + 1, idx1_v]
                    score_distance += distance_matrix_gpu[client, next_client]
                    route_demand += client_demands_global_mem[client]
                    
                # Calcul de la pénalité de capacité
                if (route_demand > vehicle_capacity_global_mem[idx1_v]):
                    capacity_violation = route_demand - vehicle_capacity_global_mem[idx1_v]
                else:
                    capacity_violation = 0
                total_penalty += lambda_ * capacity_violation

            # Calcul de la fitness initiale
            f = score_distance + total_penalty
            f_best = f

            ####### Incremental Moves

            for iter_ in range(max_iter):

                best_delta = infinit
                best_idx1_client = -1
                best_idx1_voiture = -1
                best_idx2_client = -1
                best_idx2_voiture = -1
                best_is_swap = 0

                #for idx1_v in range(nb_voitures):
                    #for idx1_c in range(1, int(size_route_global_mem[d, idx1_v] - 1)):
                    
                for current_client in range(1, nb_clients + 1):
                    
                    #current_client = current_solution_local[idx1_c, idx1_v]
                
                    idx1_c = position_clients[current_client-1,0] 
                    idx1_v = position_clients[current_client-1,1] 
                    
                    
                    if (tabuTenure[current_client - 1] <= iter_):
                        
                        ### A améliorer, peut être précalculé peut être
                        client_before = current_solution_local[idx1_c - 1, idx1_v]
                        client_after = current_solution_local[idx1_c + 1, idx1_v]
                        
                        impact_client_leaving = distance_matrix_gpu[client_before, client_after] - distance_matrix_gpu[
                            current_client, client_before] - distance_matrix_gpu[current_client, client_after]


                        if vehicle_capacity_global_mem[idx1_v] > current_demande_local[idx1_v] - \
                                client_demands_global_mem[current_client]:
                            demand_before_leaving = vehicle_capacity_global_mem[idx1_v]
                        else:
                            demand_before_leaving = current_demande_local[idx1_v] - client_demands_global_mem[
                                current_client]
                            
                            
                        if current_demande_local[idx1_v] > demand_before_leaving:
                            impact_capacite_client_leaving = current_demande_local[idx1_v] - demand_before_leaving
                        else:
                            impact_capacite_client_leaving = 0


                        #for idx2_v in range(nb_voitures):
                            #for idx2_c in range(int(size_route_global_mem[d, idx2_v] - 1)):
                        
                        
                        
                        

                        for idx2_v in range(nb_voitures):      

                            idx2_c = 0
                            new_client_before = current_solution_local[idx2_c, idx2_v]
                            
                            #new_client_before = closest_clients_gpu_memory[d,i]
                            
                            #idx2_c = position_clients[new_client_before-1,0] 
                            #idx2_v = position_clients[new_client_before-1,1]                             


                            ### Eval relocate move
                            if new_client_before != current_client and new_client_before != client_before:

                                new_client_after = current_solution_local[idx2_c + 1, idx2_v]
                                impact_distance_client_arrival = distance_matrix_gpu[new_client_before, current_client] + \
                                                        distance_matrix_gpu[current_client, new_client_after] - \
                                                        distance_matrix_gpu[new_client_before, new_client_after]



                                delta = impact_client_leaving + impact_distance_client_arrival
                                # test = delta

                                # Ajout de la pénalité liée à la capacité dans delta
                                if (idx1_v != idx2_v):
                                    
                                    # Impact de la capacité en arrivant au nouveau client
                                    if vehicle_capacity_global_mem[idx2_v] > current_demande_local[idx2_v]:
                                        demand_after_arrival = vehicle_capacity_global_mem[idx2_v]
                                    else:
                                        demand_after_arrival = current_demande_local[idx2_v]

                                    if current_demande_local[idx2_v] + client_demands_global_mem[current_client] > demand_after_arrival:
                                        
                                        impact_capacite_client_arrival = current_demande_local[idx2_v] + \
                                                                        client_demands_global_mem[
                                                                            current_client] - demand_after_arrival
                                    else:
                                        impact_capacite_client_arrival = 0
                                    
                                    delta_capacity = impact_capacite_client_arrival - impact_capacite_client_leaving
                                    delta_penalty = lambda_ * delta_capacity
                                    delta += delta_penalty
                                    # test = delta_penalty
                                # test = delta

                                # if (tabuTenure[current_client] <= iter_ or delta + f < X_best):
                                
                                if(delta < best_delta):
                                    best_delta = delta
                                    best_idx1_client = idx1_c
                                    best_idx1_voiture = idx1_v
                                    best_idx2_client = idx2_c
                                    best_idx2_voiture = idx2_v
                                    
                                    best_is_swap = 0
                                    
                                    
                        ############
                        #for i in range(nb_nearest_neighbor_tabu):        
                        
                        #for idx2_v in range(nb_voitures):
                            #for idx2_c in range(1, int(size_route_global_mem[d, idx2_v]) - 1):
                        for i in range(nb_nearest_neighbor_tabu):       
                        #for new_client_before in range(1, nb_clients + 1):
                            
                            #new_client_before = current_solution_local[idx2_c, idx2_v]
                            new_client_before = closest_clients_gpu_memory[current_client-1,i] 
                            
                            idx2_c = position_clients[new_client_before-1,0] 
                            idx2_v = position_clients[new_client_before-1,1]                             


                            ### Eval relocate move
                            if new_client_before != current_client and new_client_before != client_before:

                                new_client_after = current_solution_local[idx2_c + 1, idx2_v]
                                impact_distance_client_arrival = distance_matrix_gpu[new_client_before, current_client] + \
                                                        distance_matrix_gpu[current_client, new_client_after] - \
                                                        distance_matrix_gpu[new_client_before, new_client_after]



                                delta = impact_client_leaving + impact_distance_client_arrival
                                # test = delta

                                # Ajout de la pénalité liée à la capacité dans delta
                                if (idx1_v != idx2_v):
                                    
                                    # Impact de la capacité en arrivant au nouveau client
                                    if vehicle_capacity_global_mem[idx2_v] > current_demande_local[idx2_v]:
                                        demand_after_arrival = vehicle_capacity_global_mem[idx2_v]
                                    else:
                                        demand_after_arrival = current_demande_local[idx2_v]

                                    if current_demande_local[idx2_v] + client_demands_global_mem[current_client] > demand_after_arrival:
                                        
                                        impact_capacite_client_arrival = current_demande_local[idx2_v] + \
                                                                        client_demands_global_mem[
                                                                            current_client] - demand_after_arrival
                                    else:
                                        impact_capacite_client_arrival = 0
                                    
                                    delta_capacity = impact_capacite_client_arrival - impact_capacite_client_leaving
                                    delta_penalty = lambda_ * delta_capacity
                                    delta += delta_penalty
                                    # test = delta_penalty
                                # test = delta

                                # if (tabuTenure[current_client] <= iter_ or delta + f < X_best):
                                
                                if(delta < best_delta):
                                    best_delta = delta
                                    best_idx1_client = idx1_c
                                    best_idx1_voiture = idx1_v
                                    best_idx2_client = idx2_c
                                    best_idx2_voiture = idx2_v
                                    
                                    best_is_swap = 0
                            
                                
                                #Eval swap move
                                if (new_client_before != client_after and new_client_before != 0):
                                    
                                    
                                    client_swap = new_client_before
                                    
                                    #if (tabuTenure[client_swap - 1] <= iter_):
                                            
                                    new_client_before_after_swap = current_solution_local[idx2_c - 1, idx2_v]
                                    new_client_after = current_solution_local[idx2_c + 1, idx2_v]
                                    




                                    impact_distance_client_swap_leaving = distance_matrix_gpu[new_client_before_after_swap, current_client] - distance_matrix_gpu[
                            new_client_before_after_swap, client_swap] - distance_matrix_gpu[client_swap, current_client]
                                                        
                                                        
                                    impact_distance_client_swap_arrival = distance_matrix_gpu[client_before, client_swap] +   distance_matrix_gpu[client_swap, client_after] - distance_matrix_gpu[client_before, client_after]
                                                        
                                
                                    
                                    delta = impact_client_leaving + impact_distance_client_arrival + impact_distance_client_swap_leaving + impact_distance_client_swap_arrival
                                    


                                    if (idx1_v != idx2_v):
                                        


                                        current_demande_local_tmp = current_demande_local[idx2_v] + client_demands_global_mem[current_client] 
                                            

                                        if vehicle_capacity_global_mem[idx2_v] > current_demande_local_tmp - client_demands_global_mem[client_swap]:
                                            demand_before_leaving = vehicle_capacity_global_mem[idx2_v]
                                        else:
                                            demand_before_leaving = current_demande_local_tmp - client_demands_global_mem[ client_swap]    
                                            
                                            
                                                        
                                        if current_demande_local_tmp > demand_before_leaving:
                                            impact_capacite_client_swap_leaving = current_demande_local_tmp - demand_before_leaving
                                        else:
                                            impact_capacite_client_swap_leaving = 0   
        

                                        current_demande_local_tmp2 = current_demande_local[idx1_v] - client_demands_global_mem[int(current_client)] 


                                        if vehicle_capacity_global_mem[idx1_v] > current_demande_local_tmp2:
                                            demand_after_arrival = vehicle_capacity_global_mem[idx1_v]
                                        else:
                                            demand_after_arrival = current_demande_local_tmp2

                                        if current_demande_local_tmp2 + client_demands_global_mem[client_swap] > demand_after_arrival:
                                            
                                            impact_capacite_client_swap_arrival = current_demande_local_tmp2 + \
                                                                            client_demands_global_mem[
                                                                                client_swap] - demand_after_arrival
                                        else:
                                            impact_capacite_client_swap_arrival = 0     
                                        
                                        
                                        delta_capacity = impact_capacite_client_arrival - impact_capacite_client_leaving + impact_capacite_client_swap_arrival - impact_capacite_client_swap_leaving 
                                        
                                        
                                        delta_penalty = lambda_ * delta_capacity
                                        delta +=  delta_penalty



                                    
                                    if(delta < best_delta):
                                        best_delta = delta
                                        best_idx1_client = idx1_c
                                        best_idx1_voiture = idx1_v
                                        best_idx2_client = idx2_c
                                        best_idx2_voiture = idx2_v
                                        
                                        best_is_swap = 1                                        
                                            



                f += best_delta
                # test = f

                # Mise à jour de la current solution
                best_client = current_solution_local[best_idx1_client, best_idx1_voiture]
                
                best_client_swap = current_solution_local[best_idx2_client, best_idx2_voiture]

                best_idx1_voiture_swap = best_idx2_voiture
                best_idx2_voiture_swap = best_idx1_voiture
                
                
                if best_idx1_voiture != best_idx2_voiture:
                    
                    best_idx1_client_swap = best_idx2_client
                    best_idx2_client_swap = best_idx1_client - 1
                    
                else:
                    
                    
                    if(best_idx2_client > best_idx1_client):
                        
                        best_idx1_client_swap = best_idx2_client - 1
                        best_idx2_client_swap = best_idx1_client - 1
                        
                    else:
                        
                        best_idx1_client_swap = best_idx2_client
                        best_idx2_client_swap = best_idx1_client
                        
                        
                
                                
                if best_client != 0 and best_client != -1:
                    
                    # Mise à jour de la route 1
                    for idx1_c in range(best_idx1_client, size_route_global_mem[d, best_idx1_voiture]):
                        
                        client = current_solution_local[idx1_c + 1, best_idx1_voiture] 
                        
                        if(client != 0 and client != -1):
                            position_clients[client-1,0] = idx1_c
                            position_clients[client-1,1] = best_idx1_voiture
                        
                        current_solution_local[idx1_c, best_idx1_voiture] = current_solution_local[
                            idx1_c + 1, best_idx1_voiture]
                        
                        
                        

                    if best_idx1_voiture == best_idx2_voiture and best_idx2_client > best_idx1_client:
                        best_idx2_client = best_idx2_client - 1

                    # Mise à jour de la route 2
                    for i in range(0, size_route_global_mem[d, best_idx2_voiture] - best_idx2_client):
                        idx2_c = size_route_global_mem[d, best_idx2_voiture] - i
                        
                        client = current_solution_local[idx2_c, best_idx2_voiture] 
                        if(client != 0 and client != -1):
                            position_clients[client-1,0] = idx2_c + 1
                            position_clients[client-1,1] = best_idx2_voiture
                        
                        current_solution_local[idx2_c + 1, best_idx2_voiture] = current_solution_local[idx2_c, best_idx2_voiture]

                    
                    position_clients[best_client-1,0] = best_idx2_client + 1
                    position_clients[best_client-1,1] = best_idx2_voiture
                        
                    current_solution_local[best_idx2_client + 1, best_idx2_voiture] = best_client

                    list_tabu = int(alpha * nb_clients + int((10 * xoroshiro128p_uniform_float32(rng_states, d)) + 1))

                    tabuTenure[best_client - 1] = iter_ + list_tabu

                    # Mise à jour de la size route
                    size_route_global_mem[d, best_idx1_voiture] -= 1
                    size_route_global_mem[d, best_idx2_voiture] += 1

                    # Mise à jour de la capacite
                    current_demande_local[best_idx1_voiture] -= client_demands_global_mem[best_client]
                    current_demande_local[best_idx2_voiture] += client_demands_global_mem[best_client]



                    #Mise à jour pour le deuxième client en cas de swap
                    if(best_is_swap == 1 and best_client_swap != 0 and best_client_swap != -1):

                
                        #Mise à jour de la route 1
                        for idx1_c in range(best_idx1_client_swap, size_route_global_mem[d, best_idx1_voiture_swap]):
                            
                            client = current_solution_local[idx1_c + 1, best_idx1_voiture_swap] 
                            
                            if(client != 0 and client != -1):
                                position_clients[client-1,0] = idx1_c
                                position_clients[client-1,1] = best_idx1_voiture_swap
                        
                            current_solution_local[idx1_c, best_idx1_voiture_swap] = current_solution_local[
                                idx1_c + 1, best_idx1_voiture_swap]

                        if best_idx1_voiture_swap == best_idx2_voiture_swap and best_idx2_client_swap > best_idx1_client_swap:
                            best_idx2_client_swap = best_idx2_client_swap - 1

                        #Mise à jour de la route 2
                        for i in range(0, size_route_global_mem[d, best_idx2_voiture_swap] - best_idx2_client_swap):
                            idx2_c = size_route_global_mem[d, best_idx2_voiture_swap] - i
                            
                            client = current_solution_local[idx2_c, best_idx2_voiture_swap]
                            
                            if(client != 0 and client != -1):
                                position_clients[client-1,0] = idx2_c + 1
                                position_clients[client-1,1] = best_idx2_voiture_swap
                        
                            current_solution_local[idx2_c + 1, best_idx2_voiture_swap] = current_solution_local[
                                idx2_c, best_idx2_voiture_swap]

                        
                        position_clients[best_client_swap-1,0] = best_idx2_client_swap + 1
                        position_clients[best_client_swap-1,1] = best_idx2_voiture_swap
                    
                        current_solution_local[best_idx2_client_swap + 1, best_idx2_voiture_swap] = best_client_swap

                        list_tabu = int(alpha * nb_clients + int((10 * xoroshiro128p_uniform_float32(rng_states, d)) + 1))

                        tabuTenure[best_client_swap - 1] = iter_ + list_tabu

                        #Mise à jour de la size route
                        size_route_global_mem[d, best_idx1_voiture_swap] -= 1
                        size_route_global_mem[d, best_idx2_voiture_swap] += 1

                        #Mise à jour de la capacite
                        current_demande_local[best_idx1_voiture_swap] -= client_demands_global_mem[best_client_swap]
                        current_demande_local[best_idx2_voiture_swap] += client_demands_global_mem[best_client_swap]





                # Mise à jour de la fitness et des structures en fonction du meilleur mouvement.
                if f < f_best:

                    f_best = f
                    # Mise à jour de la meilleure solution (best_solution)

                    for idx1_v in range(nb_voitures):
                        for idx1_c in range(nb_clients + 2):
                            best_iteration_solution_local[idx1_c, idx1_v] = current_solution_local[idx1_c, idx1_v]
                    #######
                    for idx1_v in range(nb_voitures):
                        best_iteration_demande_local[idx1_v] = current_demande_local[idx1_v]

                    islegal = True
                    for idx1_v in range(nb_voitures):
                        if (current_demande_local[idx1_v] > vehicle_capacity_global_mem[idx1_v]):
                            islegal = False

                    if (islegal and f < f_best_legal):
                        f_best_legal = f

                        for idx1_v in range(nb_voitures):
                            for idx1_c in range(nb_clients + 2):
                                current_solution_global_mem[d, idx1_c, idx1_v] = current_solution_local[idx1_c, idx1_v]

                # test = f_best

            # Ajustement de lambda_penalty en fonction de la demande finale

            islegal = True
            for idx1_v in range(nb_voitures):
                if best_iteration_demande_local[idx1_v] > vehicle_capacity_global_mem[idx1_v]:
                    islegal = False

            if islegal:
                
                #lambda_ = lambda_ * 0.5
                lambda_ = lambda_ /factor_lambda
            else:
                #lambda_ = lambda_ * 2
                lambda_ = lambda_ * factor_lambda


            #if(global_iter > nb_iteration//2):
                #lambda_ = 100

    
            for idx1_v in range(nb_voitures):
                for idx1_c in range(nb_clients + 2):
                    current_solution_local[idx1_c, idx1_v] = best_iteration_solution_local[idx1_c, idx1_v]

            for idx1_v in range(nb_voitures):
                current_demande_local[idx1_v] = best_iteration_demande_local[idx1_v]

            for idx1_v in range(nb_voitures):
                size_route_global_mem[d, idx1_v] = 0

            for idx1_v in range(nb_voitures):
                for idx1_c in range(nb_clients + 2):
                    if current_solution_local[idx1_c, idx1_v] != -1:
                        size_route_global_mem[d, idx1_v] += 1

        vector_f_global_mem[d] = f_best_legal

        
