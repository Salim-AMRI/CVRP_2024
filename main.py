import argparse
import datetime
import logging

from utils.utils import read_instance, evaluate_distance_solutions, verify_solution, insertion_pop

from numba import cuda, int32
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

import numba as nb
import cuda_kernels.distance_solutions
import cuda_kernels.local_search
import cuda_kernels.init_pop
import cuda_kernels.crossover

from time import time


# Parse arguments


parser = argparse.ArgumentParser(description="Memetic algo for CVRP")

parser.add_argument("instance",  metavar='t', type=str, help="instance name")
parser.add_argument("--id_gpu", type=int, help="id_gpu", default=0)
parser.add_argument("--seed", type=int, help="seed", default=0)

parser.add_argument("--size_pop", help="size_pop", type=int, default=10000)
parser.add_argument("--alpha", help="alpha", type=float, default=0.6)

parser.add_argument("--max_iter", help="max_iter", type=int, default=1000)
parser.add_argument("--nb_iterations", help="nb_iterations", type=int, default=10)
parser.add_argument("--lambda_penalty", help="lambda_penalty", type=float, default=1)
parser.add_argument("--nb_neighbors", help="nb_neighbors", type=int, default=48)

parser.add_argument("--gamma", help="gamma", type=int, default=10)
parser.add_argument("--LS", help="LS", type=str, default="tabu_lambda")

parser.add_argument('--test', help="test", action='store_true')


args = parser.parse_args()

instance = args.instance
alpha = args.alpha
nb_neighbors = args.nb_neighbors
max_iter = args.max_iter
nb_iterations = args.nb_iterations
lambda_penalty = args.lambda_penalty
gamma = args.gamma
LS = args.LS

test = args.test
seed = args.seed

if(test):
    size_pop = 100
    max_iter = 1000
    nb_iterations = 2
    nb_neighbors = 3
else:
    size_pop = args.size_pop


print("size_pop")
print(size_pop)

name_expe = f"CVRP_{instance}_size_pop_{size_pop}_{LS}_max_iter_{max_iter}_nb_iterations_{nb_iterations}_alpha_{alpha}_lambda_penalty_{lambda_penalty}_nb_neighbors_{nb_neighbors}_gamma_{gamma}_test_{test}_seed_{seed}_{datetime.datetime.now()}.txt"


logging.basicConfig(
    handlers=[
        logging.FileHandler(f"logs/{name_expe}.log"),
        logging.StreamHandler(),
    ],
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S",
)


# Init gpu devices
cuda.select_device(args.id_gpu)
device = f"cuda:{args.id_gpu}"
logging.info(device)


# Load graph
filepath = "instances/"

nb_voitures, optimal_solution, capacity, points, client_demands = read_instance(filepath + args.instance)


# Parametre de l'algorithme
nb_clients = len(client_demands) - 1

if(nb_voitures is None):
    nb_voitures = int(instance.split(".")[0].split("k")[1])



print("nb_clients : "  + str(nb_clients))
print("nb_voitures : "  + str(nb_voitures))
print("capacity : "  + str(capacity))

min_dist_insertion = nb_clients / gamma
#min_dist_insertion = 0

vehicle_capacity = np.full(nb_voitures, capacity)


# Init global cuda variables
import cuda_kernels.distance_solutions
import cuda_kernels.local_search
import cuda_kernels.init_pop



cuda_kernels.distance_solutions.nb_clients = nb_clients
cuda_kernels.distance_solutions.nb_voitures = nb_voitures
cuda_kernels.distance_solutions.size = nb_clients + 1
cuda_kernels.init_pop.nb_clients = nb_clients
cuda_kernels.init_pop.nb_voitures = nb_voitures
cuda_kernels.local_search.size = nb_clients + 2
cuda_kernels.local_search.nb_clients = nb_clients
cuda_kernels.local_search.nb_voitures = nb_voitures
cuda_kernels.local_search.max_size_route = int(nb_clients/nb_voitures * 2)

cuda_kernels.crossover.nb_clients = nb_clients
cuda_kernels.crossover.nb_voitures = nb_voitures
cuda_kernels.crossover.max_size_route = int(nb_clients/nb_voitures * 4)


# Définir blockspergrid1 et threadsperblock
threadsperblock = 64
blockspergrid0 = (size_pop + (threadsperblock - 1)) // threadsperblock
blockspergrid1 = (size_pop * size_pop + (threadsperblock - 1)) // threadsperblock
blockspergrid2 = ((size_pop * (size_pop - 1) // 2) + (threadsperblock - 1)) // threadsperblock



# Initialiser les états du générateur de nombres aléatoires sur le GPU
rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid0, seed=int(seed))
np.random.seed(int(seed))

time_start = time()




# Calcul de la matrice de distance
distance_matrix_cvrp = np.zeros((nb_clients + 1, nb_clients + 1))

for i in range(nb_clients + 1):
    for j in range(i):
        distance_matrix_cvrp[i, j] = round(np.linalg.norm(points[i] - points[j]),0)

distance_matrix_cvrp = distance_matrix_cvrp + np.transpose(distance_matrix_cvrp)

print("distance_matrix_cvrp")
print(distance_matrix_cvrp)



# Creation des tenseurs sur le CPU
offsprings_pop = np.ones((size_pop, nb_clients + 2, nb_voitures), dtype=int) * (-1)
offsprings_pop[:, 0, :] = 0

demand_route = np.zeros((size_pop, nb_voitures), dtype=int)
size_route = np.ones((size_pop, nb_voitures), dtype=int)

fitness_pop = np.ones((size_pop), dtype=np.int32) * 999999
fitness_offsprings = np.zeros((size_pop), dtype=np.int32)

matrixDistance1 = np.zeros((size_pop, size_pop))
matrixDistance2 = np.zeros((size_pop, size_pop))
matrixDistanceAll = np.zeros((2 * size_pop, 2 * size_pop), dtype=np.int16)
matrixDistanceAll[:size_pop, :size_pop] = (np.ones((size_pop, size_pop), dtype=np.int16) * 999999)
matrice_crossovers_already_tested = np.zeros((size_pop, size_pop), dtype=np.uint8)



### Passages des données sur la carte GPU
offsprings_pop_global_mem = cuda.to_device(offsprings_pop)
size_route_global_mem = cuda.to_device(size_route)
size_route_best = np.ones((size_pop, nb_voitures), dtype=int)
size_route_best_global_mem = cuda.to_device(size_route_best)
demand_route_global_mem = cuda.to_device(demand_route)
client_demands_global_mem = cuda.to_device(client_demands)
vehicle_capacity_global_mem = cuda.to_device(vehicle_capacity)
distance_matrix_cvrp_global_mem = cuda.to_device(distance_matrix_cvrp)
fitness_offsprings_gpu_memory = cuda.to_device(fitness_offsprings)

matrixDistance1_gpu_memory = cuda.to_device(matrixDistance1)
matrixDistance2_gpu_memory = cuda.to_device(matrixDistance2)




###################################################################################################################################################

logging.info("############################")
logging.info("Init population")
logging.info("############################")
    
    

if(LS == "tabu_legal"):
    
    cuda_kernels.init_pop.generate_illegal_solutions[blockspergrid0, threadsperblock](size_pop, rng_states, demand_route_global_mem,
                                                                          client_demands_global_mem, 
                                                                          offsprings_pop_global_mem, size_route_global_mem)
    
    cuda_kernels.init_pop.repair_solutions[blockspergrid0, threadsperblock](size_pop, rng_states, distance_matrix_cvrp_global_mem,demand_route_global_mem, client_demands_global_mem, vehicle_capacity_global_mem, offsprings_pop_global_mem, size_route_global_mem)
        
elif(LS == "tabu_lambda"):
    
    cuda_kernels.init_pop.generate_illegal_solutions[blockspergrid0, threadsperblock](size_pop, rng_states,  demand_route_global_mem,
                                                                          client_demands_global_mem, offsprings_pop_global_mem, size_route_global_mem)


nb.cuda.synchronize()


solutions_pop =  offsprings_pop_global_mem.copy_to_host()



logging.info("begin verify solution init")
for i in range(size_pop):

    score, demand,  size_route, is_correct_solution = verify_solution(nb_clients, nb_voitures, distance_matrix_cvrp, client_demands, vehicle_capacity, solutions_pop[i], 0, logging)


    if(is_correct_solution != True):
        logging.info("PB solution : " + str(i))
        

logging.info("end verify solution init")   
    
    


###################################################################################################################################################

logging.info("############################")
logging.info("Start Memetic algorithm")
logging.info("############################")
    

best_score = 9999999

for epoch in range(99999):
    # First step : local search
    # Start tabu

    logging.info("############################")
    logging.info("Start TABU")
    logging.info("############################")

    startEpoch = time()
    startTabu = time()

    
    if(LS == "tabu_legal"):
    
        cuda_kernels.local_search.tabu_CVRP_legal[blockspergrid0, threadsperblock](rng_states, size_pop, max_iter, distance_matrix_cvrp_global_mem, offsprings_pop_global_mem, demand_route_global_mem, vehicle_capacity_global_mem, client_demands_global_mem, size_route_global_mem, fitness_offsprings_gpu_memory, alpha)
    
    elif(LS == "tabu_lambda"):

        cuda_kernels.local_search.tabu_CVRP_lambda[blockspergrid0, threadsperblock](rng_states, size_pop, max_iter, distance_matrix_cvrp_global_mem,
                                                       offsprings_pop_global_mem, demand_route_global_mem,
                                                       vehicle_capacity_global_mem, client_demands_global_mem,
                                                       size_route_global_mem, fitness_offsprings_gpu_memory, lambda_penalty, alpha,  nb_iterations)


    nb.cuda.synchronize()
    
    offsprings_pop = offsprings_pop_global_mem.copy_to_host()
    fitness_offsprings = fitness_offsprings_gpu_memory.copy_to_host()

   
    logging.info("############################")
    logging.info("Log results Tabu")
    logging.info("############################")     
    
    logging.info(f"Tabucol duration : {time() - startTabu}")



    best_score_pop = np.min(fitness_offsprings)
    worst_score_pop = np.max(fitness_offsprings)
    avg_pop = np.mean(fitness_offsprings)

    logging.info(f"Epoch : {epoch}")
    logging.info(
        f"Pop best : {best_score_pop}"
        f"_worst : {worst_score_pop}"
        f"_avg : {avg_pop}"
    )


    best_current_score = min(fitness_offsprings)



    if best_current_score < best_score:
        best_score = best_current_score
        logging.info("Save best solution")
        solution = offsprings_pop[
            np.argmin(fitness_offsprings)
        ]
        np.savetxt(
            f"solutions/solution_cvrp_{instance}_score_{best_current_score}_epoch_{epoch}.csv",
            solution.astype(int),
            fmt="%i",
        )

    with open("evol/" + name_expe, "a", encoding="utf8") as fichier:
        fichier.write(
            f"\n{best_score},{best_current_score},{epoch},{time() - time_start}"
        )

    # Second step : insertion of offsprings in pop according to diversity/fit criterion

    logging.info("Keep best with diversity/fit tradeoff")
    
    

    
    logging.info("############################")
    logging.info("Start distance evaluation between individuals in pop")
    logging.info("############################")     


    start_dist_eval = time()


    solutions_pop_global_mem = cuda.to_device(solutions_pop)



    cuda_kernels.distance_solutions.computeMatrixDistance_Hamming[blockspergrid1, threadsperblock](size_pop, size_pop, matrixDistance1_gpu_memory,
                                                                                 solutions_pop_global_mem, offsprings_pop_global_mem)

    matrixDistance1 = matrixDistance1_gpu_memory.copy_to_host()



    cuda_kernels.distance_solutions.computeSymmetricMatrixDistance_Hamming[blockspergrid2, threadsperblock](size_pop, matrixDistance2_gpu_memory, offsprings_pop_global_mem)

    matrixDistance2 = matrixDistance2_gpu_memory.copy_to_host()

    # Aggregate all the matrix in order to obtain a full 2*size_pop matrix with all the distances between individuals in pop and in offspring
    matrixDistanceAll[:size_pop, size_pop:] = matrixDistance1
    matrixDistanceAll[size_pop:, :size_pop] = matrixDistance1.transpose(1, 0)
    matrixDistanceAll[size_pop:, size_pop:] = matrixDistance2

    logging.info(f"Matrix distance duration : {time() - start_dist_eval}")

    logging.info("end  matrix distance")
    #####################################


    logging.info("############################")
    logging.info("Start insertion in pop")
    logging.info("############################")   
    
    start = time()

    matrixDistanceAll[:size_pop, :size_pop], fitness_pop, solutions_pop, matrice_crossovers_already_tested = insertion_pop(
        size_pop,
        matrixDistanceAll,
        solutions_pop,
        offsprings_pop,
        fitness_pop,
        fitness_offsprings,
        matrice_crossovers_already_tested,
        min_dist_insertion,
        logging
    )


    logging.info(f"Insertion in pop : {time() - start}")

    

    
    logging.info("############################")
    logging.info("Start log pop and verification")
    logging.info("############################")   



    logging.info("begin verify solution pop")
    for i in range(size_pop):

        score, demand,  size_route, is_correct_solution = verify_solution(nb_clients, nb_voitures, distance_matrix_cvrp, client_demands, vehicle_capacity, solutions_pop[i], 0, logging)


        if(is_correct_solution != True):
            logging.info("PB solution : " + str(i))
            
        if(score != fitness_pop[i] and fitness_pop[i] != 99999):
            logging.info("PB score : " + str(i))
            
        if(np.max(demand) > capacity):
            logging.info("PB demand solution " + str(i))     


                
    logging.info("end verify solution pop")     
    
    
    
    
    logging.info("After keep best info")

    best_score_pop = np.min(fitness_pop)
    worst_score_pop = np.max(fitness_pop)
    avg_score_pop = np.mean(fitness_pop)

    logging.info(
        f"Pop _best : {best_score_pop}_worst : {worst_score_pop}_avg : {avg_score_pop}"
    )
    logging.info(fitness_pop)

    matrix_distance_pop = matrixDistanceAll[:size_pop, :size_pop]

    max_dist = np.max(matrix_distance_pop)
    min_dist = np.min(matrix_distance_pop + np.eye(size_pop) * 9999)
    avg_dist = np.sum(matrix_distance_pop) / (size_pop * (size_pop - 1))

    logging.info(
        f"Avg dist : {avg_dist} min dist : {min_dist} max dist : {max_dist}"
    )

    




    # Third step : selection of best crossovers to generate new offsprings
    logging.info("############################")
    logging.info("Crossovers")
    logging.info("############################")



    start = time()
   
    
    solutions_pop_global_mem = cuda.to_device(solutions_pop)
    
    ########## Compute neighbor matching #####################
    dist_neighbors = np.where(
        matrice_crossovers_already_tested == 1,
        9999,
        matrixDistanceAll[:size_pop, :size_pop],
    )
    dist_neighbors = np.where(dist_neighbors == 0, 99999, dist_neighbors)
    closest_individuals = np.argsort(dist_neighbors, axis=1)[:, :nb_neighbors]
    select_idx = np.random.randint(nb_neighbors, size=(size_pop,1))
    rng_closest_individuals = np.take_along_axis(closest_individuals, select_idx, 1)
    closest_individuals_gpu_memory = cuda.to_device(
            np.ascontiguousarray(rng_closest_individuals[:, 0])
        )
    ###############################################"

    
    ########## Compute crossovers #####################
    cuda_kernels.crossover.computeClosestCrossover_GPX[blockspergrid1, threadsperblock](
            rng_states,
            size_pop,
            distance_matrix_cvrp_global_mem,
            solutions_pop_global_mem,
            offsprings_pop_global_mem,
            size_route_global_mem,
            demand_route_global_mem,
            client_demands_global_mem,
            closest_individuals_gpu_memory,
        )
    ###############################################"


    for i in range(size_pop):
        matrice_crossovers_already_tested[i, closest_individuals[i,0]] = 1
        

    
    logging.info(
        f"nb cross already tested in pop : {np.sum(matrice_crossovers_already_tested)}"
    )
    logging.info(f"Crossover duration : {time() - start}")

    
    
    logging.info("begin verify crosssovers")
    offspring = offsprings_pop_global_mem.copy_to_host()

    size_route_offspring = size_route_global_mem.copy_to_host()
    
    demand_route = demand_route_global_mem.copy_to_host()
    

    for i in range(size_pop):

        score, demand,  size_route, is_correct_solution = verify_solution(nb_clients, nb_voitures, distance_matrix_cvrp, client_demands, vehicle_capacity, offspring[i], 0, logging)

        
        for j in range(nb_voitures):
            if(size_route_offspring[i,j] != size_route[j]):
                print("PB size route")
                
        if(is_correct_solution != True):
            print("PB solution : " + str(i))
            

        
        for j in range(nb_voitures):
            if(demand[j] != demand_route[i,j]):
                print("PB demande : " + str(i))
                
                print(demand)
                print(demand_route[i])
                
                
    logging.info("end verify crosssovers")
    
    logging.info("end generation")

    
    logging.info(f"generation duration : {time() - startEpoch}")
    
  
