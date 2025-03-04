import numpy as np
import pandas as pd
import shapely
import geopandas as gpd
import matplotlib.pyplot as plt
from HARMONY_LUTI_SG.quantlhmodel import QUANTLHModel
from HARMONY_LUTI_SG.maps import *

def load_cij(od_matrix, intrazonal_tt, zone_codes, upper_threshold, lower_threshold):
    cij = od_matrix.copy()
    cij.columns = [int(x) for x in cij.columns]
    cij.index = [int(x) for x in cij.index]
    cij = cij.loc[zone_codes][zone_codes]
    if "ZONE_CODE" in intrazonal_tt.columns:
        intrazonal_tt = intrazonal_tt.set_index('ZONE_CODE').loc[zone_codes]
    else:
        intrazonal_tt = intrazonal_tt.loc[zone_codes]
    for idx in range(len(zone_codes)):
        cij.iloc[idx, idx] = intrazonal_tt.iloc[idx].values
    cij = cij.fillna(upper_threshold)
    cij = cij.clip(lower=lower_threshold, upper=upper_threshold)
    cij.to_numpy(dtype=float)
    return cij

def runCalculationsSimple(ODs, cijs, df_Employment, df_Residential, attractor,
                    control, convergence_criteria=0.025):
    m, n = cijs['DRIVE'].shape
    model = QUANTLHModel(m,n)
    model.setPopulationEi(df_Employment, 'ZONE_CODE', control)

    # model.setObsCbar(cbar_pt, cbar_road, cbar_bike)
    model.OBS_cbar_road = (ODs['DRIVE'] * cijs['DRIVE']).sum().sum() / ODs['DRIVE'].sum().sum()
    model.OBS_cbar_bike = (ODs['BIKE'] * cijs['BIKE']).sum().sum() / ODs['BIKE'].sum().sum()
    model.OBS_cbar_walk = (ODs['WALK'] * cijs['WALK']).sum().sum() / ODs['WALK'].sum().sum()

    # model.setODMatrix(od_pt, od_road, od_bike)
    model.OD_road = ODs['DRIVE']
    model.OD_bike = ODs['BIKE']
    model.OD_walk = ODs['WALK']

    model.cij_road = cijs['DRIVE']
    model.cij_bike = cijs['BIKE']
    model.cij_walk = cijs['WALK']

    # This is not needed - can just use cijs; but keep just in case i'm missing something
    cij_k = {}
    cij_k['DRIVE'] = model.cij_road
    cij_k['BIKE'] = model.cij_bike  # list of cost matrices
    cij_k['WALK'] = model.cij_walk  # list of cost matrices

    CBarObs = {}
    CBarObs['DRIVE'] = model.OBS_cbar_road
    CBarObs['BIKE'] = model.OBS_cbar_bike
    CBarObs['WALK'] = model.OBS_cbar_walk

    model.setAttractorsAj(df_Residential, 'ZONE_CODE', attractor)

    Beta = {k: 1 for k in ['DRIVE', 'BIKE', 'WALK']}
    iteration = 0
    converged = False
    while not converged:
        iteration += 1
        if iteration % 25 == 0:
            print("Iteration: ", iteration)

        Sij = {} # = [[] for i in range(len(cijs))]

        ExpMBetaCijk = {} # [[] for k in range(len(cijs))]
        for kk in ['DRIVE', 'BIKE', 'WALK']:
            cij_k[kk] = np.array(cij_k[kk], dtype=float)
            ExpMBetaCijk[kk] = np.exp(-Beta[kk] * cij_k[kk])

        # this is the main model loop to calculate Sij[k] trip numbers for each mode k
        for i, k in enumerate(['DRIVE', 'BIKE', 'WALK']):  # mode loop
            Sij[k] = np.zeros(model.m * model.n).reshape(model.m, model.n)
            for i in range(model.m):
                denom = 0
                for kk in ['DRIVE', 'BIKE', 'WALK']:
                    denom += np.sum(model.Aj * ExpMBetaCijk[kk][i, :])
                Sij2 = model.Ei[i] * (model.Aj * ExpMBetaCijk[k][i, :] / denom)
                Sij[k][i, :] = Sij2  # put answer slice back in return array

        converged = True
        CBarPred = {k: 0 for k in ['DRIVE', 'BIKE', 'WALK']}
        delta = {k: 1 for k in ['DRIVE', 'BIKE', 'WALK']}
        for k in ['DRIVE', 'BIKE', 'WALK']:
            CBarPred[k] = model.computeCBar(Sij[k], cij_k[k])
            delta[k] = np.absolute(CBarPred[k] - CBarObs[k])
            if delta[k] / CBarObs[k] > convergence_criteria:
                Beta[k] = Beta[k] * CBarPred[k] / CBarObs[k]
                converged = False

    return Sij, Beta, CBarPred

def luti_outputSimple(Sij, data_input, production_var, attraction_var):
    dat = data_input.copy()
    luti = dat[['ZONE_CODE', production_var, attraction_var]].copy()
    luti['_OUT_AXIS'] = Sij['DRIVE'].sum(axis=0) + Sij['BIKE'].sum(axis=0) + Sij['WALK'].sum(axis=0)
    luti['PUSHPULL'] = np.round(luti['_OUT_AXIS'] - luti[production_var])
    luti['NUM_LEAVING'] = [abs(np.round(x * 0.2)) if x < 0 else 0 for x in luti['PUSHPULL']]
    luti['NUM_RECEIVING'] = [x * 0.2 if x > 0 else 0 for x in luti['PUSHPULL']]
    luti['NUM_RECEIVING'] = [np.ceil(x * luti['NUM_LEAVING'].sum() / luti['NUM_RECEIVING'].sum()) for x in
                             luti['NUM_RECEIVING']]

    luti['NEW_POPULATION'] = luti['BASELINE_POPULATION'] + luti['NUM_RECEIVING'] + luti['NUM_LEAVING']

    luti = luti[['ZONE_CODE', 'BASELINE_POPULATION', 'BASELINE_JOBS', 'NEW_POPULATION']]

    return luti

def runCalculations(ODs, cijs, df_Employment, df_Residential, attractor = "POPULATION_SYN",
                    control = "EMPLOYMENT_SYN", convergence_criteria=0.025):
    m, n = cijs['Car'].shape
    model = QUANTLHModel(m,n)
    model.setPopulationEi(df_Employment, 'ZONE_CODE', control)

    # model.setObsCbar(cbar_pt, cbar_road, cbar_bike)
    model.OBS_cbar_road = (ODs['Car'] * cijs['Car']).sum().sum() / ODs['Car'].sum().sum()
    model.OBS_cbar_bike = (ODs['Cycle'] * cijs['Cycle']).sum().sum() / ODs['Cycle'].sum().sum()
    model.OBS_cbar_pt = (ODs['Pt'] * cijs['Pt']).sum().sum() / ODs['Pt'].sum().sum()
    model.OBS_cbar_walk = (ODs['Walk'] * cijs['Walk']).sum().sum() / ODs['Walk'].sum().sum()
    model.OBS_cbar_taxi = (ODs['Taxi'] * cijs['Taxi']).sum().sum() / ODs['Walk'].sum().sum()

    # model.setODMatrix(od_pt, od_road, od_bike)
    model.OD_road = ODs['Car']
    model.OD_bike = ODs['Cycle']
    model.OD_pt = ODs['Pt']
    model.OD_walk = ODs['Walk']
    model.OD_taxi = ODs['Taxi']

    model.cij_road = cijs['Car']
    model.cij_bike = cijs['Cycle']
    model.cij_pt = cijs['Pt']
    model.cij_walk = cijs['Walk']
    model.cij_taxi = cijs['Taxi']

    # This is not needed - can just use cijs; but keep just in case i'm missing something
    cij_k = {}
    cij_k['Car'] = model.cij_road
    cij_k['Cycle'] = model.cij_bike  # list of cost matrices
    cij_k['Pt'] = model.cij_pt
    cij_k['Walk'] = model.cij_walk  # list of cost matrices
    cij_k['Taxi'] = model.cij_taxi  # list of cost matrices

    CBarObs = {}
    CBarObs['Car'] = model.OBS_cbar_road
    CBarObs['Cycle'] = model.OBS_cbar_bike
    CBarObs['Pt'] = model.OBS_cbar_pt
    CBarObs['Walk'] = model.OBS_cbar_walk
    CBarObs['Taxi'] = model.OBS_cbar_taxi

    model.setAttractorsAj(df_Residential, 'ZONE_CODE', attractor)

    Beta = {k: 1 for k in ['Car', 'Cycle', 'Pt', 'Walk', 'Taxi']}
    iteration = 0
    converged = False
    while not converged:
        iteration += 1
        if iteration % 25 == 0:
            print("Iteration: ", iteration)

        Sij = {} # = [[] for i in range(len(cijs))]

        ExpMBetaCijk = {} # [[] for k in range(len(cijs))]
        for kk in ['Car', 'Cycle', 'Pt', 'Walk', 'Taxi']:
            cij_k[kk] = np.array(cij_k[kk], dtype=float)
            ExpMBetaCijk[kk] = np.exp(-Beta[kk] * cij_k[kk])

        # this is the main model loop to calculate Sij[k] trip numbers for each mode k
        for i, k in enumerate(['Car', 'Cycle', 'Pt', 'Walk', 'Taxi']):  # mode loop
            Sij[k] = np.zeros(model.m * model.n).reshape(model.m, model.n)
            for i in range(model.m):
                denom = 0
                for kk in ['Car', 'Cycle', 'Pt', 'Walk', 'Taxi']:
                    denom += np.sum(model.Aj * ExpMBetaCijk[kk][i, :])
                Sij2 = model.Ei[i] * (model.Aj * ExpMBetaCijk[k][i, :] / denom)
                Sij[k][i, :] = Sij2  # put answer slice back in return array

        converged = True
        CBarPred = {k: 0 for k in ['Car', 'Cycle', 'Pt', 'Walk', 'Taxi']}
        delta = {k: 1 for k in ['Car', 'Cycle', 'Pt', 'Walk', 'Taxi']}
        for k in ['Car', 'Cycle', 'Pt', 'Walk', 'Taxi']:
            CBarPred[k] = model.computeCBar(Sij[k], cij_k[k])
            delta[k] = np.absolute(CBarPred[k] - CBarObs[k])
            if delta[k] / CBarObs[k] > convergence_criteria:
                Beta[k] = Beta[k] * CBarPred[k] / CBarObs[k]
                converged = False

    return Sij, Beta, CBarPred