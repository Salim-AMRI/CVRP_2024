import numpy as np
from utils.utils import read_instance






instance = "instances/X-n106-k14.vrp"
#solution = np.loadtxt("solutions/solution_cvrp_X-n106-k14.vrp_score_26593_epoch_1.csv")

solution = np.loadtxt("test_offspring1.csv")




nb_voitures, optimal_solution, capacity, points, client_demands = read_instance(instance)


# Parametre de l'algorithme
nb_clients = len(client_demands) - 1

if(nb_voitures is None):
    nb_voitures = int(instance.split(".")[0].split("k")[1])



print("nb_clients : "  + str(nb_clients))
print("nb_voitures : "  + str(nb_voitures))
print("capacity : "  + str(capacity))


vehicle_capacity = np.full(nb_voitures, capacity)

# Calcul de la matrice de distance
distance_matrix_cvrp = np.zeros((nb_clients + 1, nb_clients + 1))

for i in range(nb_clients + 1):
    for j in range(i):
        distance_matrix_cvrp[i, j] = round(np.linalg.norm(points[i] - points[j]),0)

distance_matrix_cvrp = distance_matrix_cvrp + np.transpose(distance_matrix_cvrp)

print("distance_matrix_cvrp")
print(distance_matrix_cvrp)





print(solution)


isLegal_solution = True

list_clients = []

for i in range(nb_voitures):
    for j in range(nb_clients):
        
        
        if(solution[j,i] != 0  and solution[j,i] != -1):
            
            list_clients.append(int(solution[j,i]))
            

print(len(list_clients))

print(list_clients)

ok_client = True




for i in range(1, nb_clients + 1):
    
    if(i not in list_clients):
        
        ok_client = False
        isLegal_solution = False
        print("client " + str(i) + " is missing")
        

print("ok_client")
print(ok_client)
            
            
if(len(list_clients) != nb_clients):
    isLegal_solution = False    
    

for i in range(nb_voitures):
    
    route = []
    
    j = 0
    
    while(j < nb_clients):   

        route.append(solution[j,i])
        
        j = j + 1
    
    if(route[0] != 0 and route[-1] != 0):
        isLegal_solution = False    
        print("pb depot")




f = 0
route_demand = np.zeros((nb_voitures))


size_route = np.ones((nb_voitures))*2

for idx1_v in range(nb_voitures):
    for idx1_c in range(nb_clients + 2):

                
        if (idx1_c + 1 < nb_clients + 2):
                 
            if (solution[idx1_c + 1, idx1_v] != -1):
    
   
                
                f += distance_matrix_cvrp[int(solution[idx1_c, idx1_v]),int(solution[idx1_c + 1, idx1_v])]
                
                
                if (solution[idx1_c, idx1_v] != -1 and solution[idx1_c, idx1_v] != 0):
                    size_route[idx1_v] += 1
                    route_demand[idx1_v] += client_demands[int(solution[idx1_c, idx1_v])]
    
    
        
        
    if(route_demand[idx1_v] > vehicle_capacity[idx1_v]):
        isLegal_solution = False    
        print("problem capacity route " + str(idx1_v))
    

print("isLegal_solution")       
print(isLegal_solution)   
print("Score " + str(f))



