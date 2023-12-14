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
size2 = -1
max_clients = -1


@cuda.jit
def computeCrossover_OX(rng_states, size_pop, distance_matrix_gpu, solution_pop, offsprings_pop, size_route_offspring, demand_route_global_mem, client_demands, indices, vehicle_capacity):

    d = cuda.grid(1)
    nbParent = 2

    if d < size_pop:

        idx1 = int(d)
        idx2 = int(indices[idx1])

        client_affecte = nb.cuda.local.array((nb_clients), nb.int16)
        
        for idx1_v in range(nb_voitures):
            for idx1_c in range(size):
                offsprings_pop[d,idx1_c,idx1_v]  = -1
                
        parents = nb.cuda.local.array((nbParent, size, nb_voitures), nb.int16)

        for idx1_v in range(nb_voitures):
            for idx1_c in range(size):
                parents[0, idx1_c, idx1_v] = solution_pop[idx1, idx1_c, idx1_v]
                parents[1, idx1_c, idx1_v] = solution_pop[idx2, idx1_c, idx1_v]

        # Initialisation des vecteurs GT1 et GT2 avec des valeurs -1
        vecteur_GT1 = nb.cuda.local.array((nb_clients), dtype=np.int16)
        vecteur_GT2 = nb.cuda.local.array((nb_clients), dtype=np.int16)

        for c in range(nb_clients):
            vecteur_GT1[c] = -1
            vecteur_GT2[c] = -1

        # Initialisation des indices pour les vecteurs GT1 et GT2
        indice_GT1 = 0
        indice_GT2 = 0

        # Parcours des itinéraires de chaque ensemble de parents
        for idx_parent in range(2):  # 0 pour parents[0], 1 pour parents[1]
            for idx1_v in range(nb_voitures):
                for idx1_c in range(size):
                    client = parents[idx_parent, idx1_c, idx1_v]

                    # Vérifie si le client est entre les dépôts (client > 0)
                    if client > 0:
                        # Place le client dans le vecteur correspondant
                        if idx_parent == 0:
                            vecteur_GT1[indice_GT1] = client
                            indice_GT1 += 1
                        else:
                            vecteur_GT2[indice_GT2] = client
                            indice_GT2 += 1

        # Choix aléatoire des points de coupure
        point_de_coupure1 = int(nb_clients * xoroshiro128p_uniform_float32(rng_states, d))

        if point_de_coupure1 == nb_clients:
            point_de_coupure1 = nb_clients - 1

        # Initialisation de point_de_coupure2 pour garantir que la boucle s'exécute au moins une fois
        point_de_coupure2 = point_de_coupure1

        while point_de_coupure2 == point_de_coupure1:
            point_de_coupure2 = int(nb_clients * xoroshiro128p_uniform_float32(rng_states, d))

            if point_de_coupure2 == nb_clients:
                point_de_coupure2 = nb_clients - 1

        # Assurez-vous que point_de_coupure2 est après point_de_coupure1
        if point_de_coupure1 > point_de_coupure2:
            point_de_coupure1, point_de_coupure2 = point_de_coupure2, point_de_coupure1


        # Crée un tableau offspring de la même taille que vecteur_GT1 et initialise à -1
        offspring = nb.cuda.local.array((nb_clients), nb.int16)

        # Remplissage du tableau avec -1
        for i in range(nb_clients):
            offspring[i] = -1

        # Remplit le segment entre les points de coupure avec les clients de vecteur_GT2
        for i in range(point_de_coupure1, point_de_coupure2):
            offspring[ i] = vecteur_GT2[i]

        # Crée un vecteur intermédiaire de longueur len_GT1
        intermediate_route = nb.cuda.local.array((nb_clients), nb.int16)

        # Remplissage du tableau avec -1
        for i in range(nb_clients):
            intermediate_route[i] = -1

        idx_intermediate = 0
        for i in range(point_de_coupure2, nb_clients):
            intermediate_route[idx_intermediate] = vecteur_GT1[i]
            idx_intermediate += 1

        for i in range(point_de_coupure2):
            intermediate_route[idx_intermediate] = vecteur_GT1[i]
            idx_intermediate += 1

        for i in range(nb_clients):
            client_intermediate = intermediate_route[i]
            if client_intermediate != -1:
                for client_offspring in offspring:
                    if client_intermediate == client_offspring:
                        intermediate_route[i] = -1
                        break  # Sortir de la boucle dès qu'une correspondance est trouvée


        # Remplit le vecteur offspring avec les clients de intermediate_route différents de -1
        idx_offspring = point_de_coupure2
        for i in range(nb_clients):
            client_intermediate = intermediate_route[i]
            if client_intermediate != -1:
                offspring[idx_offspring] = client_intermediate
                idx_offspring = (idx_offspring + 1) % nb_clients

######### Splitting algorithm ########

        V = nb.cuda.local.array((size2), nb.int16)
        
        for i in range(1, nb_clients + 1):
            V[i] = 9999
            
        V[0] = 0
        
  
        P = nb.cuda.local.array((size2), nb.int16)
        for i in range(nb_clients+1):
            P[i] = -1

        for i in range(1, nb_clients + 1):
            demand_route = 0
            cost = 0
            j =  i

            while j <= nb_clients and demand_route <= vehicle_capacity[0]:
                demand_route += client_demands[int(offspring[j - 1])]

                if i == j:
                    cost = distance_matrix_gpu[0][int(offspring[j - 1])] + distance_matrix_gpu[int(offspring[j - 1])][
                        0]
                else:
                    cost = cost - distance_matrix_gpu[int(offspring[j - 2])][0] + \
                           distance_matrix_gpu[int(offspring[j - 2])][int(offspring[j - 1])] + distance_matrix_gpu[int(offspring[j - 1])][
                               0]

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
            for k in range(i+ 1, j + 1):
                
                if(t < nb_voitures):
                    offsprings_pop[d, cpt, t] =  offspring[k-1]
                    
                    client_affecte[offspring[k-1] - 1] = 1
                    
                    size_route_offspring[d, t] += 1
                    
                    demand_route_global_mem[d,t] += client_demands[offspring[k-1]]

                    cpt += 1
                
            j = i
            t = t + 1
            


        for c in range(nb_clients):
            if(client_affecte[c] == 0):
                r = int(nb_voitures * xoroshiro128p_uniform_float32(rng_states, d))
                
                if(r > nb_voitures - 1):
                    r = nb_voitures - 1
                    
                offsprings_pop[d, int(size_route_offspring[d, r]) - 1,r] = c + 1
                
                size_route_offspring[d, r] += 1
                
                demand_route_global_mem[d,r] += client_demands[c + 1]
        
        for v in range(nb_voitures):
            offsprings_pop[d,int(size_route_offspring[d, v]) - 1, v] = 0

                
@cuda.jit
def computeCrossover_AOX(rng_states, size_pop, distance_matrix_gpu, solution_pop, offsprings_pop, size_route_offspring, demand_route_global_mem, client_demands, indices, vehicle_capacity):

    d = cuda.grid(1)
    nbParent = 2

    if d < size_pop:

        idx1 = int(d)
        idx2 = int(indices[idx1])

        client_affecte = nb.cuda.local.array((nb_clients), nb.int16)
        
        for idx1_v in range(nb_voitures):
            for idx1_c in range(size):
                offsprings_pop[d,idx1_c,idx1_v]  = -1
                
        parents = nb.cuda.local.array((nbParent, size, nb_voitures), nb.int16)

        for idx1_v in range(nb_voitures):
            for idx1_c in range(size):
                parents[0, idx1_c, idx1_v] = solution_pop[idx1, idx1_c, idx1_v]
                parents[1, idx1_c, idx1_v] = solution_pop[idx2, idx1_c, idx1_v]

        # Initialisation des vecteurs GT1 et GT2 avec des valeurs -1
        vecteur_GT1 = nb.cuda.local.array((nb_clients), dtype=np.int16)
        vecteur_GT2 = nb.cuda.local.array((nb_clients), dtype=np.int16)

        for c in range(nb_clients):
            vecteur_GT1[c] = -1
            vecteur_GT2[c] = -1

        # Initialisation des indices pour les vecteurs GT1 et GT2
        indice_GT1 = 0
        indice_GT2 = 0

        # Parcours des itinéraires de chaque ensemble de parents
        for idx_parent in range(2):  # 0 pour parents[0], 1 pour parents[1]
            for idx1_v in range(nb_voitures):
                for idx1_c in range(size):
                    client = parents[idx_parent, idx1_c, idx1_v]

                    # Vérifie si le client est entre les dépôts (client > 0)
                    if client > 0:
                        # Place le client dans le vecteur correspondant
                        if idx_parent == 0:
                            vecteur_GT1[indice_GT1] = client
                            indice_GT1 += 1
                        else:
                            vecteur_GT2[indice_GT2] = client
                            indice_GT2 += 1

        # Choix aléatoire des points de coupure
        point_de_coupure1 = int(nb_clients * xoroshiro128p_uniform_float32(rng_states, d))

        if point_de_coupure1 == nb_clients:
            point_de_coupure1 = nb_clients - 1

        # Initialisation de point_de_coupure2 pour garantir que la boucle s'exécute au moins une fois
        point_de_coupure2 = point_de_coupure1

        while point_de_coupure2 == point_de_coupure1:
            point_de_coupure2 = int(nb_clients * xoroshiro128p_uniform_float32(rng_states, d))

            if point_de_coupure2 == nb_clients:
                point_de_coupure2 = nb_clients - 1

        # Assurez-vous que point_de_coupure2 est après point_de_coupure1
        if point_de_coupure1 > point_de_coupure2:
            point_de_coupure1, point_de_coupure2 = point_de_coupure2, point_de_coupure1
        
        ##### AOX Process
            
        # Récupérer le numéro du client du point de coupure du parent 2
        cut_client_number = vecteur_GT1[point_de_coupure2]

        # Trouvez la position de ce client dans le parent 1
        cut_client_position_parent1 = -1
        for i in range(len(vecteur_GT2)):
            if vecteur_GT2[i] == cut_client_number:
                cut_client_position_parent1 = i
                break
            
        #####

        # Crée un tableau offspring de la même taille que vecteur_GT1 et initialise à -1
        offspring = nb.cuda.local.array((nb_clients), nb.int16)

        # Remplissage du tableau avec -1
        for i in range(nb_clients):
            offspring[i] = -1

        # Remplit le segment entre les points de coupure avec les clients de vecteur_GT2
        for i in range(point_de_coupure1, point_de_coupure2):
            offspring[ i] = vecteur_GT2[i]

        # Crée un vecteur intermédiaire de longueur len_GT1
        intermediate_route = nb.cuda.local.array((nb_clients), nb.int16)

        # Remplissage du tableau avec -1
        for i in range(nb_clients):
            intermediate_route[i] = -1

        idx_intermediate = 0
        for i in range(cut_client_position_parent1, nb_clients):
            intermediate_route[idx_intermediate] = vecteur_GT1[i]
            idx_intermediate += 1

        for i in range(cut_client_position_parent1):
            intermediate_route[idx_intermediate] = vecteur_GT1[i]
            idx_intermediate += 1

        for i in range(nb_clients):
            client_intermediate = intermediate_route[i]
            if client_intermediate != -1:
                for client_offspring in offspring:
                    if client_intermediate == client_offspring:
                        intermediate_route[i] = -1
                        break  # Sortir de la boucle dès qu'une correspondance est trouvée


        # Remplit le vecteur offspring avec les clients de intermediate_route différents de -1
        idx_offspring = point_de_coupure2
        for i in range(nb_clients):
            client_intermediate = intermediate_route[i]
            if client_intermediate != -1:
                offspring[idx_offspring] = client_intermediate
                idx_offspring = (idx_offspring + 1) % nb_clients

######### Splitting algorithm ########

        V = nb.cuda.local.array((size2), nb.int16)
        
        for i in range(1, nb_clients + 1):
            V[i] = 9999
            
        V[0] = 0
        
  
        P = nb.cuda.local.array((size2), nb.int16)
        for i in range(nb_clients+1):
            P[i] = -1

        for i in range(1, nb_clients + 1):
            demand_route = 0
            cost = 0
            j =  i

            while j <= nb_clients and demand_route <= vehicle_capacity[0]:
                demand_route += client_demands[int(offspring[j - 1])]

                if i == j:
                    cost = distance_matrix_gpu[0][int(offspring[j - 1])] + distance_matrix_gpu[int(offspring[j - 1])][
                        0]
                else:
                    cost = cost - distance_matrix_gpu[int(offspring[j - 2])][0] + \
                           distance_matrix_gpu[int(offspring[j - 2])][int(offspring[j - 1])] + distance_matrix_gpu[int(offspring[j - 1])][
                               0]

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
            for k in range(i+ 1, j + 1):
                
                if(t < nb_voitures):
                    offsprings_pop[d, cpt, t] =  offspring[k-1]
                    
                    client_affecte[offspring[k-1] - 1] = 1
                    
                    size_route_offspring[d, t] += 1
                    
                    demand_route_global_mem[d,t] += client_demands[offspring[k-1]]

                    cpt += 1
                
            j = i
            t = t + 1


        for c in range(nb_clients):
            if(client_affecte[c] == 0):
                r = int(nb_voitures * xoroshiro128p_uniform_float32(rng_states, d))
                
                if(r > nb_voitures - 1):
                    r = nb_voitures - 1
                    
                offsprings_pop[d, int(size_route_offspring[d, r]) - 1,r] = c + 1
                
                size_route_offspring[d, r] += 1
                
                demand_route_global_mem[d,r] += client_demands[c + 1]
        
        for v in range(nb_voitures):
            offsprings_pop[d,int(size_route_offspring[d, v]) - 1, v] = 0
            

@cuda.jit
def computeCrossover_LOX(rng_states, size_pop, distance_matrix_gpu, solution_pop, offsprings_pop, size_route_offspring, demand_route_global_mem, client_demands, indices, vehicle_capacity):

    d = cuda.grid(1)
    nbParent = 2

    if d < size_pop:

        idx1 = int(d)
        idx2 = int(indices[idx1])

        client_affecte = nb.cuda.local.array((nb_clients), nb.int16)
        
        for idx1_v in range(nb_voitures):
            for idx1_c in range(size):
                offsprings_pop[d,idx1_c,idx1_v]  = -1
                
        parents = nb.cuda.local.array((nbParent, size, nb_voitures), nb.int16)

        for idx1_v in range(nb_voitures):
            for idx1_c in range(size):
                parents[0, idx1_c, idx1_v] = solution_pop[idx1, idx1_c, idx1_v]
                parents[1, idx1_c, idx1_v] = solution_pop[idx2, idx1_c, idx1_v]

        # Initialisation des vecteurs GT1 et GT2 avec des valeurs -1
        vecteur_GT1 = nb.cuda.local.array((nb_clients), dtype=np.int16)
        vecteur_GT2 = nb.cuda.local.array((nb_clients), dtype=np.int16)

        for c in range(nb_clients):
            vecteur_GT1[c] = -1
            vecteur_GT2[c] = -1

        # Initialisation des indices pour les vecteurs GT1 et GT2
        indice_GT1 = 0
        indice_GT2 = 0

        # Parcours des itinéraires de chaque ensemble de parents
        for idx_parent in range(2):  # 0 pour parents[0], 1 pour parents[1]
            for idx1_v in range(nb_voitures):
                for idx1_c in range(size):
                    client = parents[idx_parent, idx1_c, idx1_v]

                    # Vérifie si le client est entre les dépôts (client > 0)
                    if client > 0:
                        # Place le client dans le vecteur correspondant
                        if idx_parent == 0:
                            vecteur_GT1[indice_GT1] = client
                            indice_GT1 += 1
                        else:
                            vecteur_GT2[indice_GT2] = client
                            indice_GT2 += 1

        # Choix aléatoire des points de coupure
        point_de_coupure1 = int(nb_clients * xoroshiro128p_uniform_float32(rng_states, d))

        if point_de_coupure1 == nb_clients:
            point_de_coupure1 = nb_clients - 1

        # Initialisation de point_de_coupure2 pour garantir que la boucle s'exécute au moins une fois
        point_de_coupure2 = point_de_coupure1

        while point_de_coupure2 == point_de_coupure1:
            point_de_coupure2 = int(nb_clients * xoroshiro128p_uniform_float32(rng_states, d))

            if point_de_coupure2 == nb_clients:
                point_de_coupure2 = nb_clients - 1

        # Assurez-vous que point_de_coupure2 est après point_de_coupure1
        if point_de_coupure1 > point_de_coupure2:
            point_de_coupure1, point_de_coupure2 = point_de_coupure2, point_de_coupure1


        # Crée un tableau offspring de la même taille que vecteur_GT1 et initialise à -1
        offspring = nb.cuda.local.array((nb_clients), nb.int16)

        # Remplissage du tableau avec -1
        for i in range(nb_clients):
            offspring[i] = -1

        # Remplit le segment entre les points de coupure avec les clients de vecteur_GT2
        for i in range(point_de_coupure1, point_de_coupure2):
            offspring[ i] = vecteur_GT2[i]

        # Crée un vecteur intermédiaire de longueur len_GT1
        intermediate_route = nb.cuda.local.array((nb_clients), nb.int16)

        # Remplissage du tableau avec -1
        for i in range(nb_clients):
            intermediate_route[i] = -1
            
        ##### LOX Process

        idx_intermediate = 0
        for i in range(nb_clients):
            intermediate_route[idx_intermediate] = vecteur_GT1[i]
            idx_intermediate += 1

        for i in range(nb_clients):
            client_intermediate = intermediate_route[i]
            if client_intermediate != -1:
                for client_offspring in offspring:
                    if client_intermediate == client_offspring:
                        intermediate_route[i] = -1
                        break
                    
        # Remplit le vecteur offspring avec les clients de intermediate_route différents de -1
        for i in range(len(offspring)):
            if offspring[i] == -1:
                for j in range(len(intermediate_route)):
                    if intermediate_route[j] != -1:
                        # Remplacer la valeur dans offspring
                        offspring[i] = intermediate_route[j]
                        intermediate_route[j] = -1
                        break
            

######### Splitting algorithm ########

        V = nb.cuda.local.array((size2), nb.int16)
        
        for i in range(1, nb_clients + 1):
            V[i] = 9999
            
        V[0] = 0
        
  
        P = nb.cuda.local.array((size2), nb.int16)
        for i in range(nb_clients+1):
            P[i] = -1

        for i in range(1, nb_clients + 1):
            demand_route = 0
            cost = 0
            j =  i

            while j <= nb_clients and demand_route <= vehicle_capacity[0]:
                demand_route += client_demands[int(offspring[j - 1])]

                if i == j:
                    cost = distance_matrix_gpu[0][int(offspring[j - 1])] + distance_matrix_gpu[int(offspring[j - 1])][
                        0]
                else:
                    cost = cost - distance_matrix_gpu[int(offspring[j - 2])][0] + \
                           distance_matrix_gpu[int(offspring[j - 2])][int(offspring[j - 1])] + distance_matrix_gpu[int(offspring[j - 1])][
                               0]

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
            for k in range(i+ 1, j + 1):
                
                if(t < nb_voitures):
                    offsprings_pop[d, cpt, t] =  offspring[k-1]
                    
                    client_affecte[offspring[k-1] - 1] = 1
                    
                    size_route_offspring[d, t] += 1
                    
                    demand_route_global_mem[d,t] += client_demands[offspring[k-1]]

                    cpt += 1
                
            j = i
            t = t + 1


        for c in range(nb_clients):
            if(client_affecte[c] == 0):
                r = int(nb_voitures * xoroshiro128p_uniform_float32(rng_states, d))
                
                if(r > nb_voitures - 1):
                    r = nb_voitures - 1
                    
                offsprings_pop[d, int(size_route_offspring[d, r]) - 1,r] = c + 1
                
                size_route_offspring[d, r] += 1
                
                demand_route_global_mem[d,r] += client_demands[c + 1]
        
        for v in range(nb_voitures):
            offsprings_pop[d,int(size_route_offspring[d, v]) - 1, v] = 0
            
        
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
            
            
    
    
    
@cuda.jit
def compute_pathRelinking(rng_states, size_pop, distance_matrix_gpu, solution_pop, offspring, size_route_offspring, demand_route_global_mem, client_demands, indices):


    d = cuda.grid(1)
    nbParent = 2

    if d < size_pop:

        idx1 = int(d)
        idx2 = int(indices[idx1])

        parents = nb.cuda.local.array((nbParent, size, nb_voitures), nb.int16)

        for idx1_v in range(nb_voitures):
            for idx1_c in range(nb_clients + 2):
                parents[0, idx1_c, idx1_v] = solution_pop[idx1, idx1_c, idx1_v]
                parents[1, idx1_c, idx1_v] = solution_pop[idx2, idx1_c, idx1_v]


        for idx1_v in range(nb_voitures):
            for idx1_c in range(nb_clients + 2):
                offspring[d,idx1_c,idx1_v] = -1
                
        for idx1_v in range(nb_voitures):
            size_route_offspring[d, idx1_v] = 1
            demand_route_global_mem[d, idx1_v] = 0
            
            offspring[d,0,idx1_v] = 0


        A = nb.cuda.local.array((nb_voitures,nb_voitures), nb.int16)


        affectation = nb.cuda.local.array((nb_voitures), nb.int16)


        D = nb.cuda.local.array((nb_clients, nb_clients), dtype=nb.int16)


        size_route1 = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        size_route2 = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        
        for id_v in range(nb_voitures):
            size_route1[id_v] = 0
            size_route2[id_v] = 0
            
        for id_v in range(nb_voitures):
            for id_c in range(nb_clients + 2):
                if parents[0, id_c, id_v] != -1:
                    size_route1[id_v] += 1
                if parents[1, id_c, id_v] != -1:
                    size_route2[id_v] += 1
        
        
        copy = nb.cuda.local.array((nb_clients), dtype=nb.int16)

        arr_all_move = nb.cuda.local.array((nb_voitures,nb_clients, 2), nb.int16 )    

        arr_moving_clients = nb.cuda.local.array((nb_clients), nb.int16 )    

        size_moves= nb.cuda.local.array((nb_voitures), nb.int16 )    
        
                    
        for r1 in range(nb_voitures):
            for r2 in range(nb_voitures):


                n = int(size_route1[r1]) - 2
                m = int(size_route2[r2]) - 2
                
                for i in range(n+1):
                    for j in range(m+1):
                        D[i, j] = 0





                for i in range(n + 1):
                    D[i, 0] = i

                for j in range(m + 1):
                    D[0, j] = j

                for i in range(1, n + 1):
                    for j in range(1, m + 1):

                        if (parents[0, i, r1]  == parents[1, j, r2]):

                            c = 0

                        else:

                            c = 9999

                        test = min(D[i - 1, j] + 1, D[i, j - 1] + 1)
                        D[i, j] = min(test, D[i - 1, j - 1] + c)

                d1 = D[n, m]

                for i in range(n+1):
                    for j in range(m+1):
                        D[i, j] = 0


                for i in range(n + 1):
                    D[i, 0] = i

                for j in range(m + 1):
                    D[0, j] = j

                for i in range(1, n + 1):
                    for j in range(1, m + 1):

                        if (parents[0, i, r1] == parents[1, m - j + 1, r2]):

                            c = 0

                        else:

                            c = 9999

                        test = min(D[i - 1, j] + 1, D[i, j - 1] + 1)
                        D[i, j] = min(test, D[i - 1, j - 1] + c)

                d2 = D[n, m]

                if(d1 < d2):
                    A[r1, r2] = d1
                else:
                    A[r1, r2] = d2
                                
                    for j in range(1, m + 1):
                        copy[j] = parents[1,m - j + 1,r2]

                    for j in range(1, m + 1):
                        parents[1,j,r2] = copy[j]
                    

        for c in range(nb_voitures):

            minVal = 99999
            minI = -1
            minJ = -1

            for i in range(nb_voitures):

                for j in range(nb_voitures):

                    if (A[i, j] < minVal):
                        minVal = A[i, j]

                        minI = i
                        minJ = j

            for k in range(nb_voitures):
                A[minI, k] = 9999
                A[k, minJ] = 9999

            affectation[minI] = minJ
            
    
            


        for k in range(nb_voitures):

            cpt_move = 0

            r1 = k
            r2 = int(affectation[k])


            n = size_route1[r1] -2 
            m = size_route2[r2] -2

            for i in range(n+1):
                for j in range(m+1):
                    D[i, j] = 0
                        
                        
            for i in range(n + 1):
                D[i, 0] = i

            for j in range(m + 1):
                D[0, j] = j

            for i in range(1, n + 1):
                for j in range(1, m + 1):

                    if (parents[0,i, r1] == parents[1,j, r2]):

                        c = 0

                    else:

                        c = 9999

                    test = min(D[i - 1, j] + 1, D[i, j - 1] + 1)
                    D[i, j] = min(test, D[i - 1, j - 1] + c)

            d1 = D[n, m]
    
            pos1 = n
            pos2 = m
    
            cpt = 0
            while (pos1 != 0 or pos2!= 0):
            #for cpt in range(10):
    
                current_val = D[pos1, pos2 ]

                if(pos2 - 1 >= 0 and D[pos1 , pos2-1] == current_val  - 1 ):


                    arr_all_move[k,cpt_move,0] = 0
                    arr_all_move[k, cpt_move, 1] = parents[1,pos2, r2]
                    cpt_move += 1

                    size_moves[k] += 1

                    arr_moving_clients[parents[1,pos2, r2] - 1] = 1

                    pos2 = pos2 - 1



                elif(pos1 - 1 >= 0  and D[pos1  - 1, pos2] == current_val - 1):

   

                    arr_all_move[k,cpt_move,0] = 1
                    arr_all_move[k, cpt_move, 1] = parents[0,pos1, r1]
                    cpt_move += 1
                    size_moves[k] += 1



                    arr_moving_clients[parents[0,pos1, r1] - 1] = 1

                    pos1 = pos1 - 1


                elif(pos1 - 1  >= 0  and pos2 - 1 >= 0   and D[pos1 -1 , pos2 - 1] == current_val ):


                    arr_all_move[k,cpt_move,0] = 2
                    arr_all_move[k, cpt_move, 1] = parents[1,pos2, r2]
                    cpt_move += 1
                    size_moves[k] += 1


                    pos1 = pos1 - 1
                    pos2 = pos2 - 1
                    
                #cpt += 1


        for c in range(nb_clients):

            if(arr_moving_clients[c] == 1):
                r = xoroshiro128p_uniform_float32(rng_states, d)
                if(r < 0.5 ):
                    arr_moving_clients[c] = 0
                
            

        for idx_v in range(nb_voitures):

            cpt = 1
            idx_c = 1

            for i in range(int(size_moves[idx_v])):

                idx_move = int(size_moves[idx_v]) - 1 - i

                motif = int(arr_all_move[idx_v, idx_move, 0])
                client = int(arr_all_move[idx_v, idx_move, 1])


                if(motif == 1 ):

                    if(arr_moving_clients[client - 1] == 1):
                        
                        if(client != -1):
                            offspring[d,idx_c, idx_v] = client
                            demand_route_global_mem[d, idx_v] += client_demands[client]
                            idx_c += 1
                            size_route_offspring[d,idx_v] += 1     
                            

                elif (motif == 0 ):

                    if(arr_moving_clients[client - 1] == 1):
                        cpt += 1

                    else:
                        
                        if(parents[1,cpt, int(affectation[idx_v])] != -1):
                            offspring[d,idx_c, idx_v] = parents[1,cpt, int(affectation[idx_v])]
                            demand_route_global_mem[d, idx_v] += client_demands[offspring[d,idx_c, idx_v]]
                            size_route_offspring[d,idx_v] += 1
                            idx_c += 1
                            cpt += 1
                            
                            

                else:

                    if(parents[1,cpt, int(affectation[idx_v])] != -1):
                        offspring[d,idx_c, idx_v] = parents[1,cpt, int(affectation[idx_v])]
                        demand_route_global_mem[d, idx_v] += client_demands[offspring[d,idx_c, idx_v]]
                        size_route_offspring[d,idx_v] += 1
                        idx_c += 1
                        cpt += 1
                        


        for idx_v in range(nb_voitures):
            offspring[d,int(size_route_offspring[d,idx_v]), idx_v] = 0

            size_route_offspring[d,idx_v] += 1                
                    
    
