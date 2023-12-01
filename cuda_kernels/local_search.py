import numba as nb
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32


nb_clients = -1
nb_voitures = -1
size = -1
max_size_route = -1

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

                                    delta = impact_client_leaving + impact_client_arrival
                                    # test = delta

                                    # Ajout de la pénalité liée à la capacité dans delta
                                    if (idx1_v != idx2_v):
                                        delta_capacity = impact_capacite_client_arrival - impact_capacite_client_leaving
                                        delta_penalty = lambda_ * delta_capacity
                                        delta += delta_penalty
                                        # test = delta_penalty
                                    # test = delta

                                    # if (tabuTenure[current_client] <= iter_ or delta + f < X_best):
                                    if ((tabuTenure[current_client - 1] <= iter_) or ((delta + f) < f_best)) and (
                                            delta < best_delta):
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
def tabu_CVRP_lambda_with_swap(rng_states, D, max_iter, distance_matrix_gpu, current_solution_global_mem,
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

                                    delta = impact_client_leaving + impact_client_arrival
                                    # test = delta

                                    # Ajout de la pénalité liée à la capacité dans delta
                                    if (idx1_v != idx2_v):
                                        delta_capacity = impact_capacite_client_arrival - impact_capacite_client_leaving
                                        delta_penalty = lambda_ * delta_capacity
                                        delta += delta_penalty
                                        # test = delta_penalty
                                    # test = delta

                                    # if (tabuTenure[current_client] <= iter_ or delta + f < X_best):
                                    if ((tabuTenure[current_client - 1] <= iter_) or ((delta + f) < f_best)) and (
                                            delta < best_delta):
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
