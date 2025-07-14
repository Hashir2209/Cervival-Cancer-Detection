import random
import numpy
import math
from solution import solution
import time
import transfer_functions_benchmark
import fitnessFUNs

def GWO(objf,lb,ub,dim,SearchAgents_no,Max_iter,trainInput,trainOutput):
    # initialize alpha, beta, and delta positions
    Alpha_pos = numpy.zeros(dim)
    Alpha_score = float("inf")
    
    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")
    
    Delta_pos = numpy.zeros(dim)
    Delta_score = float("inf")
    
    # Initialize positions with binary values
    Positions = numpy.random.randint(2, size=(SearchAgents_no,dim))
    
    Convergence_curve1 = numpy.zeros(Max_iter)
    Convergence_curve2 = numpy.zeros(Max_iter)
    
    s = solution()
    
    print("\nGWO is optimizing  \""+objf.__name__+"\"")    
    
    timerStart = time.time() 
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Main loop
    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):
            # Clip positions to boundaries
            Positions[i,:] = numpy.clip(Positions[i,:], lb, ub)
            
            # Ensure at least one feature is selected
            while numpy.sum(Positions[i,:]) == 0:   
                Positions[i,:] = numpy.random.randint(2, size=(1,dim))
            
            # Calculate fitness
            fitness = objf(Positions[i,:],trainInput,trainOutput,dim)
            
            # Update Alpha, Beta, and Delta
            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos = Positions[i,:].copy()
            elif fitness < Beta_score:
                Beta_score = fitness
                Beta_pos = Positions[i,:].copy()
            elif fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i,:].copy()
        
        # Linear decrease from 2 to 0
        a = 2 - l * ((2)/Max_iter)
        
        # Update positions
        for i in range(0,SearchAgents_no):
            current_wolf = Positions[i,:].copy()
            current_fitness = objf(current_wolf,trainInput,trainOutput,dim)
            
            # Standard GWO position update
            X_GWO = numpy.zeros(dim)
            for j in range(dim):
                # Alpha
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - current_wolf[j])
                temp = transfer_functions_benchmark.s1(A1 * D_alpha)
                X1 = Alpha_pos[j] + temp if random.random() < temp else Alpha_pos[j]
                
                # Beta
                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - current_wolf[j])
                temp = transfer_functions_benchmark.s1(A2 * D_beta)
                X2 = Beta_pos[j] + temp if random.random() < temp else Beta_pos[j]
                
                # Delta
                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - current_wolf[j])
                temp = transfer_functions_benchmark.s1(A3 * D_delta)
                X3 = Delta_pos[j] + temp if random.random() < temp else Delta_pos[j]
                
                X_GWO[j] = (X1 + X2 + X3) / 3
            
            # i-GWO enhancement
            # Compute R (neighborhood radius)
            R = current_fitness - objf(X_GWO,trainInput,trainOutput,dim)
            
            # Build neighborhood
            neighborhood = []
            for k in range(SearchAgents_no):
                neighbor_fitness = objf(Positions[k,:],trainInput,trainOutput,dim)
                if current_fitness - neighbor_fitness <= R:
                    neighborhood.append(Positions[k,:])
            
            # Handle empty neighborhood
            if len(neighborhood) == 0:
                distances = [(current_fitness - objf(Positions[k,:],trainInput,trainOutput,dim), k) 
                           for k in range(SearchAgents_no)]
                closest_idx = min(distances, key=lambda x: x[0])[1]
                neighborhood.append(Positions[closest_idx,:])
            
            # Generate DLH solution
            X_DLH = numpy.zeros(dim)
            for j in range(dim):
                random_neighbor = random.choice(neighborhood)
                random_wolf = Positions[random.randint(0, SearchAgents_no-1),:]
                X_DLH[j] = current_wolf[j] + random.random() * (random_neighbor[j] - random_wolf[j])
            
            # Binary conversion for X_DLH
            for j in range(dim):
                if random.random() < transfer_functions_benchmark.s1(X_DLH[j]):
                    X_DLH[j] = 1
                else:
                    X_DLH[j] = 0
            
            # Select better solution between X_GWO and X_DLH
            gwo_fitness = objf(X_GWO,trainInput,trainOutput,dim)
            dlh_fitness = objf(X_DLH,trainInput,trainOutput,dim)
            
            if gwo_fitness < dlh_fitness:
                candidate = X_GWO
            else:
                candidate = X_DLH
            
            # Update position if better
            if objf(candidate,trainInput,trainOutput,dim) < current_fitness:
                Positions[i,:] = candidate
        
        # Update convergence curves
        featurecount = numpy.sum(Alpha_pos)
        Convergence_curve1[l] = Alpha_score
        Convergence_curve2[l] = featurecount
        
        if (l%1==0):
            print(['At iteration'+ str(l+1)+' the best fitness on training is:'+ str(Alpha_score)+', the best number of features: '+str(featurecount)])
    
    timerEnd = time.time()  
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.bestIndividual = Alpha_pos
    s.convergence1 = Convergence_curve1
    s.convergence2 = Convergence_curve2
    s.optimizer = "GWO"
    s.objfname = objf.__name__
    
    return s
