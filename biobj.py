import pandas as pd
import numpy as np
import pickle
import geopandas as gpd
from gurobipy import Model, GRB, LinExpr, quicksum


def read_data():
    filename_hist = 'data/franklin_hist.csv'
    hist = pd.read_csv(filename_hist)

    # block to tract
    hist['TRACT'] = hist['GEOID10'].astype(str).str[:11]
    col_names = hist.columns.to_numpy()
    col_names = np.delete(col_names, [0, -1])
    hist = hist.groupby('TRACT').sum()[col_names]
    hist = hist.reset_index()

    # HHGQ (8) $*$ VOTINGAGE (2) $*$ HISPANIC (2) $*$ CENRACE (63) to VOTINGAGE (2) $*$ HISPANIC (2) $*$ RACE (7)
    n2, n3, n4 = 2, 2, 63
    for y in range(n2):  # voting age
        y = '{number:0{width}d}'.format(width=2, number=y)
        col_names = [col for col in hist.columns if y in col[2:4] and len(col)==8]

        for z in range(n3):  # ethnicity
            z = '{number:0{width}d}'.format(width=2, number=z)
            col_names2 = [col for col in col_names if z in col[4:6]]

            col_two_or_more_races = []
            for x in range(n4):  # race
                if x >= 0 and x <= 5:
                    x = '{number:0{width}d}'.format(width=2, number=x)
                    col_names3 = [col for col in col_names2 if x in col[6:8]]
                    hist[x + y + z] = hist[col_names3].sum(axis=1)
                else:
                    x = '{number:0{width}d}'.format(width=2, number=x)
                    col_names3 = [col for col in col_names2 if x in col[6:8]]
                    col_two_or_more_races.extend(col_names3)
            hist['06' + y + z] = hist[col_two_or_more_races].sum(axis=1)
    hist.drop([col for col in hist.columns if len(col)==8], axis=1, inplace=True)
    return hist


def aggregate(hist):
    # HISPANIC (2) $*$ RACE (7)
    n3, n4 = 2, 7
    hist2 = hist.copy()
    for x in range(n3):  # ethnicity
        x = '{number:0{width}d}'.format(width=2, number=x)
        for y in range(n4):  # race
            y = '{number:0{width}d}'.format(width=2, number=y)
            col_names = [col for col in hist2.columns if x in col[4:6] and y in col[0:2]]
            hist2[y + x] = hist2[col_names].sum(axis=1)
    hist2.drop([col for col in hist2.columns if len(col)==6], axis=1, inplace=True)

    # RACE (7)
    n4 = 7
    hist3 = hist.copy()
    for y in range(n4):  # race
        y = '{number:0{width}d}'.format(width=2, number=y)
        col_names = [col for col in hist3.columns if y in col[0:2]]
        hist3[y] = hist3[col_names].sum(axis=1)
    hist3.drop([col for col in hist3.columns if len(col)==6], axis=1, inplace=True)
    return hist2, hist3


def inputs(hist, r=3):
    # define all the input data for the model
    I, K = hist.shape[0], hist.shape[1] - 1
    
    V = []
    for k in range(1, K+1):
        listElem = []
        for l in range(1, r+1):
            listElem.append(hist.index[hist.iloc[:,k] == l].tolist())
        V.append(listElem)

    A = hist.iloc[:,1:].to_numpy()
    # print(A.shape, A[0])

    W = np.zeros([I, I, K])
    for i in range(I):
        for j in range(I):
            for k in range(K):
                if A[i, k] == 0 or A[j, k] == 0:
                    W[i, j, k] = 50
                else:
                    W[i, j, k] = 1 / A[i, k] + 1 / A[j, k]
    
    C = [1 / ((l + 1)**2) for l in range(r)]
    return I, K, V, A, W, C            


def coverage_1(I, K, V, r):
    ## define coverage aijk
    T = np.ones((I, I, K))

    for i in range(I): 
        for j in range(I):
            for k in range(K):
                for l in range(r):
                    if i == j or j in V[k][l]:
                        T[i, j, k] = 0
    return T


def risks(I, K, V, A, m, r, hist, hist2, hist3):
    # VOTINGAGE (2) $*$ HISPANIC (2) $*$ RACE (7)
    theta_all = np.zeros([I, I, K])
    for k in range(K):
        for l in range(r):
            for i in range(I):
                if i not in V[k][l] and A[i, k] != 0:
                    theta_all[i, i, k] = 1
    for var in m.getVars():
        name = var.VarName.split("_")
        theta_all[int(name[1]), int(name[2]), int(name[3])] = var.X

    p = np.ones([I, K])
    for k in range(K):
        for j in range(I):
            sum = 0 
            for i in range(I):
                sum += theta_all[i, j, k] * A[i, k]
            p[j, k] = theta_all[j, j, k] / sum
    p[~np.isfinite(p)] = 0
    print("Identification prob (VA*E*R): ", np.sum(p) / p.size)
    tau1 = np.sum(p) / p.size

    v = 0
    for k in range(K):
        for i in range(I):
            if A[i, k] == 1 and theta_all[i, i, k] == 1:
                v += 1
    print("Unique prob (VA*E*R): ", np.sum(v) / (I * K))
    phi1 = np.sum(v) / (I * K)


    # HISPANIC (2) $*$ RACE (7)
    K2 = hist2.shape[1] - 1
    Q2 = np.zeros([K2, K])
    for idx2, col2 in enumerate(hist2.iloc[:,1:].columns):
        x = col2[0:2]
        y = col2[2:4]
        for idx, col in enumerate(hist.iloc[:,1:].columns):
            if x == col[0:2] and y == col[4:6]:
                Q2[idx2, idx] = 1
            else:
                Q2[idx2, idx] = 0
    theta2_all = np.zeros([I, I, K2])
    for k2 in range(K2):
        for i in range(I):
            for j in range(I):
                sum1, sum2 = 0, 0
                for k in range(K):
                    sum1 += Q2[k2, k] * theta_all[i, j, k] * A[i, k]
                    sum2 += Q2[k2, k] * A[i, k]
                theta2_all[i, j, k2] = sum1 / sum2                
    theta2_all[~np.isfinite(theta2_all)] = 0

    p = np.ones([I, K2])
    for k2 in range(K2):
        for j in range(I):
            sum = 0
            for k in range(K):
                for i in range(I):
                    sum += Q2[k2, k] * theta_all[i, j, k] * A[i, k]
            p[j, k2] = theta2_all[j, j, k2] / sum
    p[~np.isfinite(p)] = 0
    print("Identification prob (E*R): ", np.sum(p) / p.size)
    tau2 = np.sum(p) / p.size

    v = 0
    A2 = hist2.iloc[:,1:].to_numpy()
    for k2 in range(K2):
        for i in range(I):
            if A2[i, k2] == 1 and theta2_all[i, i, k2] == 1:
                v += 1
    print("Unique prob (E*R): ", np.sum(v) / (I * K2))
    phi2 = np.sum(v) / (I * K2)


    # RACE (7)
    K3 = hist3.shape[1] - 1
    Q3 = np.zeros([K3, K])
    for idx3, col3 in enumerate(hist3.iloc[:,1:].columns):
        x = col3[0:2]
        for idx, col in enumerate(hist.iloc[:,1:].columns):
            if x == col[0:2]:
                Q3[idx3, idx] = 1
            else:
                Q3[idx3, idx] = 0
    theta3_all = np.zeros([I, I, K3])
    for k3 in range(K3):
        for i in range(I):
            for j in range(I):
                sum1, sum2 = 0, 0
                for k in range(K):
                    sum1 += Q3[k3, k] * theta_all[i, j, k] * A[i, k]
                    sum2 += Q3[k3, k] * A[i, k]
                theta3_all[i, j, k3] = sum1 / sum2                
    theta3_all[~np.isfinite(theta3_all)] = 0 

    p = np.ones([I, K3])
    for k3 in range(K3):
        for j in range(I):
            sum = 0
            for k in range(K):
                for i in range(I):
                    sum += Q3[k3, k] * theta_all[i, j, k] * A[i, k]
            p[j, k3] = theta3_all[j, j, k3] / sum
    p[~np.isfinite(p)] = 0
    print("Identification prob (R): ", np.sum(p) / p.size) 
    tau3 = np.sum(p) / p.size

    v = 0
    A3 = hist3.iloc[:,1:].to_numpy()
    for k3 in range(K3):
        for i in range(I):
            if A3[i, k3] == 1 and theta3_all[i, i, k3] == 1:
                v += 1
    print("Unique prob: ", np.sum(v) / (I * K3))
    phi3 = np.sum(v) / (I * K3)

    return tau1, tau2, tau3, phi1, phi2, phi3, K2, K3, theta_all, theta2_all, theta3_all, A2, A3


def smape(I, K, K2, K3, A, A2, A3, theta_all, theta2_all, theta3_all):
    # VOTINGAGE (2) $*$ HISPANIC (2) $*$ RACE (7)
    delta = np.zeros([I, K])
    for k in range(K):
        for j in range(I):
            new = 0
            for i in range(I):
                new += theta_all[i, j, k] * A[i, k]
            delta[j, k] = abs(A[j, k] - new) / (A[j, k] + new)
    delta[~np.isfinite(delta)] = 0
    print("SMAPE: ", np.sum(delta) / (I * K))
    smape1 = np.sum(delta) / (I * K)

    # HISPANIC (2) $*$ RACE (7)
    delta2 = np.zeros([I, K2])
    for k2 in range(K2):
        for j in range(I):
            new = 0
            for i in range(I):
                new += theta2_all[i, j, k2] * A2[i, k2]
            delta2[j, k2] = abs(A2[j, k2] - new) / (A2[j, k2] + new)
    delta2[~np.isfinite(delta2)] = 0
    print("SMAPE: ", np.sum(delta2) / (I * K2))
    smape2 = np.sum(delta2) / (I * K2)

    # RACE (7)
    delta3 = np.zeros([I, K3])
    for k3 in range(K3):
        for j in range(I):
            new = 0
            for i in range(I):
                new += theta3_all[i, j, k3] * A3[i, k3]
            delta3[j, k3] = abs(A3[j, k3] - new) / (A3[j, k3] + new)
    delta3[~np.isfinite(delta3)] = 0
    print("SMAPE: ", np.sum(delta3) / (I * K3))
    smape3 = np.sum(delta3) / (I * K3)

    return smape1, smape2, smape3


def payoff(I, K, V, T, A, W, C, nj, r):
    # initialize model
    m1 = Model('td')
    # m.Params.LogToConsole = 0

    # add objective function
    obj1, obj2 = LinExpr(), LinExpr()

    # add decision variables and objective function
    theta = {}
    for k in range(K):
        for l in range(r):
            if len(V[k][l]) == 0:
                continue
            for i in V[k][l]:
                for j in range(I):
                    # decision variables
                    theta[i, j, k] = m1.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="theta_%d_%d_%d"%(i, j, k))
                    # objective
                    obj1 += C[l] * T[i, j, k] * theta[i, j, k] * A[i, k]
                    obj2 += T[i, j, k] * theta[i, j, k] * A[i, k] * W[i, j, k]
                    
                # add constraints
                m1.addConstr(quicksum(theta[i, j, k] for j in range(I)) == 1)
                m1.addConstr(theta[i, i, k] + quicksum(T[i, j, k] * theta[i, j, k] for j in range(I)) == 1)

    for j in range(I):
        m1.addConstr(quicksum(quicksum(quicksum(T[i, j, k] * theta[i, j, k] * A[i, k] for i in V[k][l]) for l in range(r)) for k in range(K)) <= nj)

    m1.setObjectiveN(-obj1, index=0, priority=2, name="privacy")
    m1.setObjectiveN(obj2, index=1, priority=1, name="utility")

    m1.update()
    m1.optimize()
    m1.params.ObjNumber = 1
    f2_max = m1.ObjNVal
    return f2_max


def run_model(epsilon, I, K, T, A, W, V, r):
    # initialize model
    m = Model('td')

    # add objective function
    obj = LinExpr()

    # add decision variables and objective function
    theta = {}
    for k in range(K):
        for l in range(r):
            if len(V[k][l]) == 0:
                continue
            for i in V[k][l]:
                for j in range(I):
                    # decision variables
                    theta[i, j, k] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="theta_%d_%d_%d"%(i, j, k))
                    # objective
                    obj += C[l] * T[i, j, k] * theta[i, j, k] * A[i, k]
                    
                # add constraints
                m.addConstr(quicksum(theta[i, j, k] for j in range(I)) == 1)
                m.addConstr(theta[i, i, k] + quicksum(T[i, j, k] * theta[i, j, k] for j in range(I)) == 1)

    for j in range(I):
        m.addConstr(quicksum(quicksum(quicksum(T[i, j, k] * theta[i, j, k] * A[i, k] for i in V[k][l]) for l in range(r)) for k in range(K)) <= nj)

    # m.addConstr(quicksum(quicksum(quicksum(quicksum(C[l] * T[i, j, k] * theta[i, j, k] * A[i, k] for i in V[k][l]) for l in range(r)) for k in range(K)) for j in range(I)) >= epsilon)

    m.addConstr(quicksum(quicksum(quicksum(quicksum(T[i, j, k] * theta[i, j, k] * A[i, k] * W[i, j, k] for i in V[k][l]) for l in range(r)) for k in range(K)) for j in range(I)) <= epsilon)

    m.setObjective(obj, GRB.MAXIMIZE)

    m.update()
    m.optimize()
    return m



# define inputs
nj = 20
r = [1, 2, 3]
intervals = 20

hist = read_data()
hist2, hist3 = aggregate(hist)

with open('bi_objective.csv', 'w') as fw:
    fw.write('lambda,f1,f2,predicate,risk_1,risk_2,smape\n')
    fw.flush()

    # lexicographic optimization
    for i in r:
        I, K, V, A, W, C = inputs(hist, r=i)
        T = coverage_1(I, K, V, i)
        f2_max = payoff(I, K, V, T, A, W, C, nj, i)
        print("f2_max: ", f2_max)
        for j in range(intervals + 1):
            epsilon = f2_max * j / intervals
            print("Epsilon: ", epsilon)
            m = run_model(epsilon, I, K, T, A, W, V, i)
            f1 = m.getObjective().getValue()

            tau1, tau2, tau3, phi1, phi2, phi3, K2, K3, theta_all, theta2_all, theta3_all, A2, A3 = risks(I, K, V, A, m, i, hist, hist2, hist3)
            smape1, smape2, smape3 = smape(I, K, K2, K3, A, A2, A3, theta_all, theta2_all, theta3_all)

            fw.write(str(i) + ',' + str(f1) + ',' + str(epsilon) + ',' + "VER" + ',' + str(tau1) + ',' + str(phi1) + ',' + str(smape1) + '\n')
            fw.write(str(i) + ',' + str(f1) + ',' + str(epsilon) + ',' + "ER" + ',' + str(tau2) + ',' + str(phi2) + ',' + str(smape2) + '\n')
            fw.write(str(i) + ',' + str(f1) + ',' + str(epsilon) + ',' + "R" + ',' + str(tau3) + ',' + str(phi3) + ',' + str(smape3) + '\n')
            fw.flush()

            # theta
            filename = 'bi_obj_sols/lambda' + str(i) + '_eps' + str(j) + '.pickle'
            pickle.dump(theta_all, open(filename, "wb"))


# for i in r:
#     I, K, V, A, W = inputs(hist, r=i)
#     T = coverage_1(I, K, V)
#     f1_max = payoff(I, K, V, T, A, W, nj)
#     print("f1_max: ", f1_max)
#     for j in range(intervals + 1):
#         filename = 'data/bi_obj_sols/lambda' + str(i) + '_eps' + str(j) + '.pickle'
#         epsilon = f1_max * j / intervals
#         print("Epsilon: ", epsilon)
#         m = run_model(epsilon, I, K, T, A, W, V)
        
#         theta_all = np.zeros([I, I, K])
#         for k0 in range(K):
#             for i0 in range(I):
#                 if i0 not in V[k0] and A[i0, k0] != 0:
#                     theta_all[i0, i0, k0] = 1
#         for var in m.getVars():
#             name = var.VarName.split("_")
#             theta_all[int(name[1]), int(name[2]), int(name[3])] = var.X
        
#         pickle.dump(theta_all, open(filename, "wb"))



# tracts = gpd.read_file("data\\franklin_tract10.shp")
# tracts = tracts.to_crs(epsg=4326)
# tracts.head()

# for x in r:
#     I, K, V, A, W = inputs(hist, r=x)
#     for y in range(intervals + 1):
#         theta = pickle.load(open("data\\bi_obj_sols\\lambda" + str(x) + "_eps" + str(y) + ".pickle", "rb"))
#         with open("data\\bi_obj_sols\\lambda" + str(x) + "_eps" + str(y) + ".csv", 'w') as fw:
#             fw.write('start_id,start_geoid,start_lat,start_lon,end_id,end_geoid,end_lat,end_lon,density\n')
#             fw.flush()
            
#             for i in range(theta.shape[0]):
#                 for j in range(theta.shape[1]):
#                     if i != j:
#                         density = 0
#                         for k in range(theta.shape[2]):
#                             if theta[i, j, k] != 0:
#                                 density += theta[i, j, k] * A[i, k]
                                
#                                 start_geoid = hist.loc[i]["TRACT"]
#                                 start_geom = tracts.loc[tracts["GEOID10"] == start_geoid]
#                                 start_lat, start_lon = start_geom["INTPTLAT10"].values[0], start_geom["INTPTLON10"].values[0]
                                
#                                 end_geoid = hist.loc[j]["TRACT"]
#                                 end_geom = tracts.loc[tracts["GEOID10"] == end_geoid]
#                                 end_lat, end_lon = end_geom["INTPTLAT10"].values[0], end_geom["INTPTLON10"].values[0]

#                                 fw.write(str(i) + ',' + str(start_geoid) + ',' + str(start_lat) + ',' + str(start_lon) + ',' + str(j) + ',' + str(end_geoid) + ',' + str(end_lat) + ',' + str(end_lon) + ',' + str(density) + '\n')
#                                 fw.flush()