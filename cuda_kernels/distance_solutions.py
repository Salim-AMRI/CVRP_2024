
import math
import numba as nb
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numba import cuda


nb_voitures = -1
nb_clients = -1
size = -1
max_size_route = -1

# CUDA kernel : compute symmetric distance matrix between solutions
@cuda.jit
def computeSymmetricMatrixDistance_Hamming(size_pop, matrixDistance, solution_pop):

    d = cuda.grid(1)
    if d < (size_pop * (size_pop - 1) / 2):
        # Get upper triangular matrix indices from thread index !
        idx1 = int( size_pop - 2 - int( math.sqrt(-8.0 * d + 4.0 * size_pop * (size_pop - 1) - 7) / 2.0 - 0.5 ) )
        idx2 = int( d   + idx1   + 1  - size_pop * (size_pop - 1) / 2 + (size_pop - idx1) * ((size_pop - idx1) - 1) / 2 )


        M1 = nb.cuda.local.array((size,size), dtype=nb.int8)
        M2 = nb.cuda.local.array((size,size), dtype=nb.int8)

            
        for id_v in range(nb_voitures):
            for id_c in range( max_size_route):
                if solution_pop[idx1, id_c + 1, id_v] != -1:
                    M1[solution_pop[idx1, id_c , id_v], solution_pop[idx1, id_c + 1, id_v]] = 1
                       
                if solution_pop[idx2, id_c + 1, id_v] != -1:
                    M2[solution_pop[idx2, id_c , id_v], solution_pop[idx2, id_c + 1, id_v]] = 1
                    
        distance = 0
        
        for i in range(size):
            for j in range(size):
                
                if(M1[i,j] != M2[i,j]):
                    
                    distance += 1
                    
                    
        matrixDistance[int(idx1), int(idx2)] = distance
        matrixDistance[int(idx2), int(idx1)] = distance
        
        
    

# CUDA kernel : compute symmetric distance matrix between solutions
@cuda.jit
def computeSymmetricMatrixDistance_Hamming_v2(size_pop, matrixDistance, solution_pop):

    d = cuda.grid(1)
    if d < (size_pop * (size_pop - 1) / 2):
        # Get upper triangular matrix indices from thread index !
        idx1 = int( size_pop - 2 - int( math.sqrt(-8.0 * d + 4.0 * size_pop * (size_pop - 1) - 7) / 2.0 - 0.5 ) )
        idx2 = int( d   + idx1   + 1  - size_pop * (size_pop - 1) / 2 + (size_pop - idx1) * ((size_pop - idx1) - 1) / 2 )


        M1 = nb.cuda.local.array((nb_clients,2), dtype=nb.int8)
        M2 = nb.cuda.local.array((nb_clients,2), dtype=nb.int8)

        
            
        for id_v in range(nb_voitures):
            for id_c in range( max_size_route):
                if solution_pop[idx1, id_c + 1, id_v] != -1 and solution_pop[idx1, id_c + 1, id_v] != 0:
                    
                    M1[solution_pop[idx1, id_c , id_v] - 1 , 0] = solution_pop[idx1, id_c - 1, id_v]
                    M1[solution_pop[idx1, id_c , id_v] - 1, 1] = solution_pop[idx1, id_c + 1, id_v]
                       
                if solution_pop[idx2, id_c + 1, id_v] != -1 and solution_pop[idx2, id_c + 1, id_v] != 0:
                    
                    M1[solution_pop[idx2, id_c , id_v] - 1, 0] = solution_pop[idx2, id_c - 1, id_v]
                    M1[solution_pop[idx2, id_c , id_v] - 1, 1] = solution_pop[idx2, id_c + 1, id_v]
               
            
        distance = 0
        
        for i in range(nb_clients):

            if(M1[i,0] != M2[i,0]):
                    
                    distance += 1

            if(M1[i,1] != M2[i,1]):
                    
                    distance += 1
                    
                    
        matrixDistance[int(idx1), int(idx2)] = distance//2
        matrixDistance[int(idx2), int(idx1)] = distance//2
        


# Cuda kernel to compute distance between existing pop solution and affspring solutions
@cuda.jit
def computeMatrixDistance_Hamming_v2(size_sub_pop, size_sub_pop2, matrixDistance, tSolution1, tSolution2):

    d = cuda.grid(1)
    if d < size_sub_pop * size_sub_pop2:

        idx1 = int(d // size_sub_pop2)
        idx2 = int(d % size_sub_pop2)

        M1 = nb.cuda.local.array((nb_clients,2), dtype=nb.int8)
        M2 = nb.cuda.local.array((nb_clients,2), dtype=nb.int8)

        
            
        for id_v in range(nb_voitures):
            for id_c in range( max_size_route):
                if tSolution1[idx1, id_c + 1, id_v] != -1 and tSolution1[idx1, id_c + 1, id_v] != 0:
                    
                    M1[tSolution1[idx1, id_c , id_v] - 1 , 0] = tSolution1[idx1, id_c - 1, id_v]
                    M1[tSolution1[idx1, id_c , id_v] - 1, 1] = tSolution1[idx1, id_c + 1, id_v]
                       
                if tSolution2[idx2, id_c + 1, id_v] != -1 and tSolution2[idx2, id_c + 1, id_v] != 0:
                    
                    M1[tSolution2[idx2, id_c , id_v] - 1, 0] = tSolution2[idx2, id_c - 1, id_v]
                    M1[tSolution2[idx2, id_c , id_v] - 1, 1] = tSolution2[idx2, id_c + 1, id_v]
               
            
        distance = 0
        
        for i in range(nb_clients):

            if(M1[i,0] != M2[i,0]):
                    
                    distance += 1

            if(M1[i,1] != M2[i,1]):
                    
                    distance += 1
                    
                    
        matrixDistance[int(idx1), int(idx2)] = distance//2
        
        
        
# CUDA kernel : compute symmetric distance matrix between solutions
@cuda.jit
def computeSymmetricMatrixDistance_Sorensen(size_pop, matrixDistance, solution_pop):

    d = cuda.grid(1)
    if d < (size_pop * (size_pop - 1) / 2):
        # Get upper triangular matrix indices from thread index !
        idx1 = int( size_pop - 2 - int( math.sqrt(-8.0 * d + 4.0 * size_pop * (size_pop - 1) - 7) / 2.0 - 0.5 ) )
        idx2 = int( d   + idx1   + 1  - size_pop * (size_pop - 1) / 2 + (size_pop - idx1) * ((size_pop - idx1) - 1) / 2 )

        A = nb.cuda.local.array((nb_voitures, nb_voitures), dtype=nb.int16)

        D = nb.cuda.local.array((nb_clients, nb_clients), dtype=nb.int16)


        size_route1 = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        size_route2 = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        
        for id_v in range(nb_voitures):
            size_route1[id_v] = 0
            size_route2[id_v] = 0
            
        for id_v in range(nb_voitures):
            for id_c in range(nb_clients + 2):
                if solution_pop[idx1, id_c, id_v] != -1 and solution_pop[idx1, id_c, id_v] != 0:
                    size_route1[id_v] += 1
                if solution_pop[idx2, id_c, id_v] != -1 and solution_pop[idx2, id_c, id_v] != 0:
                    size_route2[id_v] += 1
                    
                    
        for r1 in range(nb_voitures):
            for r2 in range(nb_voitures):


                n = size_route1[r1]
                m = size_route1[r2]
                
                for i in range(n+1):
                    for j in range(m+1):
                        D[i, j] = 0





                for i in range(n + 1):
                    D[i, 0] = i

                for j in range(m + 1):
                    D[0, j] = j

                for i in range(1, n + 1):
                    for j in range(1, m + 1):

                        if (solution_pop[idx1, i, r1 ] == solution_pop[idx2, j, r2 ]):

                            c = 0

                        else:

                            c = 99999

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

                        if (solution_pop[idx1, i, r1] == solution_pop[idx2, m - j + 1, r2]):

                            c = 0

                        else:

                            c = 99999

                        test = min(D[i - 1, j] + 1, D[i, j - 1] + 1)
                        D[i, j] = min(test, D[i - 1, j - 1] + c)

                d2 = D[n, m]

                if(d1 < d2):
                    A[r1, r2] = d1
                else:
                    A[r1, r2] = d2

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

            for k in range(nb_voitures):
                A[minI, k] = 9999
                A[k, minJ] = 9999

            distance += minVal



        matrixDistance[int(idx1), int(idx2)] = distance/2
        matrixDistance[int(idx2), int(idx1)] = distance/2




        
        

# Cuda kernel to compute distance between existing pop solution and affspring solutions
@cuda.jit
def computeMatrixDistance_Hamming(size_sub_pop, size_sub_pop2, matrixDistance, tSolution1, tSolution2):

    d = cuda.grid(1)
    if d < size_sub_pop * size_sub_pop2:

        idx1 = int(d // size_sub_pop2)
        idx2 = int(d % size_sub_pop2)

        M1 = nb.cuda.local.array((size,size), dtype=nb.int8)
        M2 = nb.cuda.local.array((size,size), dtype=nb.int8)

            
        for id_v in range(nb_voitures):
            for id_c in range(max_size_route):
                if tSolution1[idx1, id_c + 1, id_v] != -1:
                    M1[tSolution1[idx1, id_c , id_v], tSolution1[idx1, id_c + 1, id_v]] = 1
                       
                if tSolution2[idx2, id_c + 1, id_v] != -1:
                    M2[tSolution2[idx2, id_c , id_v], tSolution2[idx2, id_c + 1, id_v]] = 1
                    
        distance = 0
        
        for i in range(size):
            for j in range(size):
                
                if(M1[i,j] != M2[i,j]):
                    
                    distance += 1


        matrixDistance[int(idx1), int(idx2)] = distance/2
        
        
        
# Cuda kernel to compute distance between existing pop solution and affspring solutions
@cuda.jit
def computeMatrixDistance_Sorensen(size_sub_pop, size_sub_pop2, matrixDistance, tSolution1, tSolution2):

    d = cuda.grid(1)
    if d < size_sub_pop * size_sub_pop2:

        idx1 = int(d // size_sub_pop2)
        idx2 = int(d % size_sub_pop2)

        A = nb.cuda.local.array((nb_voitures, nb_voitures), dtype=nb.int16)

        D = nb.cuda.local.array((nb_clients, nb_clients), dtype=nb.int16)


        size_route1 = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        size_route2 = nb.cuda.local.array((nb_voitures), dtype=nb.int16)
        
        for id_v in range(nb_voitures):
            size_route1[id_v] = 0
            size_route2[id_v] = 0
            
        for id_v in range(nb_voitures):
            for id_c in range(nb_clients + 2):
                if tSolution1[idx1, id_c, id_v] != -1 and tSolution1[idx1, id_c, id_v] != 0:
                    size_route1[id_v] += 1
                if tSolution2[idx2, id_c, id_v] != -1 and tSolution2[idx2, id_c, id_v] != 0:
                    size_route2[id_v] += 1
                    

        for r1 in range(nb_voitures):
            for r2 in range(nb_voitures):

                for i in range(nb_clients):
                    for j in range(nb_clients):
                        D[i, j] = 0

                n = size_route1[r1]
                m = size_route2[r2]



                for i in range(n + 1):
                    D[i, 0] = i

                for j in range(m + 1):
                    D[0, j] = j

                for i in range(1, n + 1):
                    for j in range(1, m + 1):

                        if (tSolution1[idx1, i, r1 ] == tSolution2[idx2, j, r2 ]):

                            c = 0

                        else:

                            c = 99999

                        test = min(D[i - 1, j] + 1, D[i, j - 1] + 1)
                        D[i, j] = min(test, D[i - 1, j - 1] + c)

                d1 = D[n, m]

                for i in range(nb_clients):
                    for j in range(nb_clients):
                        D[i, j] = 0


                for i in range(n + 1):
                    D[i, 0] = i

                for j in range(m + 1):
                    D[0, j] = j

                for i in range(1, n + 1):
                    for j in range(1, m + 1):

                        if (tSolution1[idx1, i, r1] == tSolution2[idx2, m - j + 1, r2]):

                            c = 0

                        else:

                            c = 99999

                        test = min(D[i - 1, j] + 1, D[i, j - 1] + 1)
                        D[i, j] = min(test, D[i - 1, j - 1] + c)

                d2 = D[n, m]

                if(d1 < d2):
                    A[r1, r2] = d1
                else:
                    A[r1, r2] = d2

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

            for k in range(nb_voitures):
                A[minI, k] = 9999
                A[k, minJ] = 9999

            distance += minVal


        matrixDistance[int(idx1), int(idx2)] = distance/2
