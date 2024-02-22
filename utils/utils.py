import re
import numpy as np


def read_instance(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    nb_voitures = None
    optimal_solution = None
    capacity = None
    points = []
    client_demands = []

    reading_coords = False
    reading_demands = False

    for line in lines:
        line = line.strip()

        if line.startswith("NODE_COORD_SECTION"):
            reading_coords = True
            reading_demands = False
        elif line.startswith("DEMAND_SECTION"):
            reading_coords = False
            reading_demands = True
        elif line.startswith("COMMENT"):
            match_num_vehicles = re.search(r'No of trucks: (\d+)', line)
            match_optimal_solution = re.search(r'Optimal value: (\d+)', line)

            if match_num_vehicles:
                nb_voitures = int(match_num_vehicles.group(1))

            if match_optimal_solution:
                optimal_solution = int(match_optimal_solution.group(1))

        elif line.startswith("CAPACITY"):
            #match_vehicle_capacity = re.search(r'CAPACITY :  (\d+)', line)

            #if match_vehicle_capacity:
            capacity = int(line.split(":")[1])

        elif line.startswith("DEPOT_SECTION"):
            break

        elif reading_coords and re.match(r'\d+\s+\d+\s+\d+', line):
            parts = re.split(r'\s+', line)
            points.append((int(parts[1]), int(parts[2])))

        elif reading_demands and re.match(r'\d+\s+-?\d+', line):
            parts = re.split(r'\s+', line)
            client_demands.append(int(parts[1]))

    print(nb_voitures, optimal_solution, capacity, np.array(points), np.array(client_demands))


    return nb_voitures, optimal_solution, capacity, np.array(points), np.array(client_demands)


###########

def verify_solution(nb_clients, nb_voitures, distance_matrix, client_demands, vehicule_capacity, current_solution, lambda_, logging):
    f = 0
    route_demand = np.zeros((nb_voitures))
    nb_clients_in_solution = 0
    
    size_route = np.ones((nb_voitures))*2

    is_correct_solution = True

    for idx1_v in range(nb_voitures):
        # print("voiture " + str(idx1_v))
        
        if(current_solution[0, idx1_v] != 0):
            is_correct_solution = False
            logging.info("pb first depot")
        
        nb_depot = 0
        
        for idx1_c in range(nb_clients + 2):
            
            if (current_solution[idx1_c , idx1_v] == 0):
                nb_depot += 1
                    
            if (idx1_c + 1 < nb_clients + 2):
                

                    
                if (current_solution[idx1_c + 1, idx1_v] != -1):

                    f += distance_matrix[current_solution[idx1_c, idx1_v]][current_solution[idx1_c + 1, idx1_v]]
                    
                    
                    if (current_solution[idx1_c, idx1_v] != -1 and current_solution[idx1_c, idx1_v] != 0):
                        nb_clients_in_solution += 1
                        
                        size_route[idx1_v] += 1
                        route_demand[idx1_v] += client_demands[current_solution[idx1_c, idx1_v]]
        
        
        if(current_solution[int(size_route[idx1_v]) - 1, idx1_v] != 0):
            is_correct_solution = False
            logging.info("pb last depot")
            
        if(nb_depot != 2):
            is_correct_solution = False
            logging.info("nb depots")
            
            
        if(route_demand[idx1_v] > vehicule_capacity[idx1_v]):
           
           f += lambda_ * (route_demand[idx1_v] - vehicule_capacity[idx1_v])


                        
                        
    if(nb_clients_in_solution != nb_clients):
        logging.info("nb clients")
        is_correct_solution = False
    

    return f, route_demand, size_route, is_correct_solution




def evaluate_distance_solutions(nb_voitures, solution1, solution2, size_route1, size_route2):


    A = np.zeros((nb_voitures, nb_voitures))

    for r1 in range(nb_voitures):
        for r2 in range(nb_voitures):
            # A[i,j] = min(dist_route(s1[i], s2[j]), dist_route(s1[i], s2[j][::-1]) )

            route_1 = solution1[1:(size_route1[r1] - 1), r1]
            route_2 = solution2[1:(size_route2[r2] - 1), r2]

            # d1 = dist_route(route_1, route_2)

            n = route_1.shape[0]
            m = route_2.shape[0]

            D = np.zeros((n + 1, m + 1))

            for i in range(n + 1):
                D[i, 0] = i

            for j in range(m + 1):
                D[0, j] = j

            for i in range(1, n + 1):
                for j in range(1, m + 1):

                    if (route_1[i - 1] == route_2[j - 1]):

                        c = 0

                    else:

                        c = 99999

                    test = min(D[i - 1, j] + 1, D[i, j - 1] + 1)
                    D[i, j] = min(test, D[i - 1, j - 1] + c)

            d1 = D[n, m]

            route_2_prim = route_2[::-1]


            n = route_1.shape[0]
            m = route_2_prim.shape[0]

            D = np.zeros((n + 1, m + 1))

            for i in range(n + 1):
                D[i, 0] = i

            for j in range(m + 1):
                D[0, j] = j

            for i in range(1, n + 1):
                for j in range(1, m + 1):

                    if (route_1[i - 1] == route_2_prim[j - 1]):

                        c = 0

                    else:

                        c = 99999

                    test = min(D[i - 1, j] + 1, D[i, j - 1] + 1)
                    D[i, j] = min(test, D[i - 1, j - 1] + c)

            d2 = D[n, m]


            A[r1, r2] = min(d1, d2)


    distance = 0


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

        A[minI, :] = 9999
        A[:, minJ] = 9999

        distance += minVal


    return distance/2





def insertion_pop(
    size_pop,
    matrixDistanceAll,
    colors_pop,
    offsprings_pop_after_tabu,
    fitness_pop,
    fitness_offsprings_after_tabu,
    matrice_crossovers_already_tested,
    min_dist,
    logging
):


    all_scores = np.hstack((fitness_pop, fitness_offsprings_after_tabu))
    matrice_crossovers_already_tested_new = np.zeros(
        (size_pop * 2, size_pop * 2), dtype=np.uint8
    )
    matrice_crossovers_already_tested_new[
        :size_pop, :size_pop
    ] = matrice_crossovers_already_tested
    idx_best = np.argsort(all_scores)
    idx_selected = []
    cpt = 0
    for i in range(0, size_pop * 2):
        idx = idx_best[i]
        if len(idx_selected) > 0:
            dist = np.min(matrixDistanceAll[idx, idx_selected])
        else:
            dist = 9999
        if dist >= min_dist:
            idx_selected.append(idx)
            if idx >= size_pop:
                cpt += 1
        if len(idx_selected) == size_pop:
            break
    logging.info(f"len(idx_selected) {len(idx_selected)}")
    if len(idx_selected) != size_pop:
        for i in range(0, size_pop * 2):
            idx = idx_best[i]
            if idx not in idx_selected:
                dist = np.min(matrixDistanceAll[idx, idx_selected])
                if dist >= 0:
                    idx_selected.append(idx)
            if len(idx_selected) == size_pop:
                break
    logging.info(f"Nb insertion {cpt}")
    new_matrix = matrixDistanceAll[idx_selected, :][:, idx_selected]
    stack_all = np.vstack((colors_pop, offsprings_pop_after_tabu))
    colors_pop_v2 = stack_all[idx_selected]
    fitness_pop_v2 = all_scores[idx_selected]
    matrice_crossovers_already_tested_v2 = matrice_crossovers_already_tested_new[
        idx_selected, :
    ][:, idx_selected]

    return new_matrix, fitness_pop_v2, colors_pop_v2, matrice_crossovers_already_tested_v2
