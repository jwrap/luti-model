"""
HARMONY Land-Use Transport-Interaction Model - Athens case study
main.py

Author: Fulvio D. Lopane, Centre for Advanced Spatial Analysis, University College London
https://www.casa.ucl.ac.uk

- Developed from Richard Milton's QUANT_RAMP
- Further developed from Eleni Kalantzi's code for MSc dissertation
Msc Smart Cities and Urban Analytics, Centre for Advanced Spatial Analysis, University College London
"""


import random
import time
from geojson import dump
import numpy as np
import pandas as pd
import shapely
import os
from statistics import mean

from HARMONY_LUTI_SG.globals import *
from HARMONY_LUTI_SG.analytics import graphProbabilities, flowArrowsGeoJSON
from HARMONY_LUTI_SG.quantlhmodel import QUANTLHModel
from HARMONY_LUTI_SG.maps import *

import csv

Flows = False

def start_main(inputs, outputs, logger):
    ############################################
    # Initialisation
    ############################################

    # NOTE: this section provides the base data for the models that come later. This
    # will only be run on the first run of the program to assemble all the tables
    # required from the original sources. After that, if the file exists in the
    # directory, then nothing new is created and this section is effectively
    # skipped, up until the model run section at the end.

    # make a model-runs dir if we need it
    if not os.path.exists(modelRunsDir):
        os.makedirs(modelRunsDir)

    ############################################
    # Model Run Section
    ############################################

    logger.warning("Importing SG cij matrices")

    zonecodes_SG = pd.read_csv(inputs["ZoneCodeFile"])
    zonecodes_SG.set_index('SUB_MTZ_NO')
    zonecodes_SG_list = zonecodes_SG['SUB_MTZ_NO'].tolist()

    # dfcoord = pd.read_csv(inputs["ZoneCoordinates"], usecols = ['zone', 'E_KOORD','N_KOORD'], index_col= 'zone') # DELETE

    # Import cij data

    intrazone_dist_df = pd.read_csv(inputs["IntrazoneDist"], usecols=["Intrazone_Dist"])
    intrazone_dist_list = intrazone_dist_df["Intrazone_Dist"].values.tolist()

    # Import csv cij private as Pandas Df
    cij_pr_df = pd.read_csv(inputs["Cij_base_roads"], header=None)

    # Convert the dataframe to numpy matrix:
    cij_pr = cij_pr_df.to_numpy()
    cij_pr[cij_pr < 0] = 120 # upper limit is 2h, -1 values need to be changed -> DO IN PREPROCESSING
    cij_pr[cij_pr < 1] = 1 # no link should be below 1

    # Change values in the main diagonal from 0 to intrazone impedance
    av_speed_pr = 30 # define avg speed in km/h CHECK for SG vehicles
    average_TT_pr = [] # average intrazonal TT
    for i in intrazone_dist_list:
        average_TT_pr.append((i / 1000) / ( av_speed_pr / 60 )) # result in minutes
    np.fill_diagonal(cij_pr, average_TT_pr) # save average TT as main diag

    # Print the dim of the matrix to check
    logger.warning('cij private shape:' + str(cij_pr.shape))
    cij_pr[cij_pr < 1] = 1 # repeat of above to remove zeros on diag
    # cij_pr = cij_pr * 2.5 # only do if times not correct, should however be since from OD Generator

    # Import the csv cij public as Pandas
    cij_pu_df = pd.read_csv(inputs["Cij_base_pt"], header=None, sep = ";")

    # Convert the dataframe to a numpy matrix:
    cij_pu = cij_pu_df.to_numpy()
    # Convert cij_pu to float64
    cij_pu = cij_pu.astype(np.float64)

    cij_pu[cij_pu == -1] = 99999.0
    cij_pu[cij_pu < 0] = 120.0 # upper limit
    cij_pu[cij_pu < 1] = 1.0 # set min
    waiting_time_proportion = 3.5 / 35.53569 # 3.5 avg wait time, 35.xxx average travel time
    cij_pu[cij_pu< 9999] *= (1 + waiting_time_proportion)

    # Change value in the diag from 0 to intrazone impedance
    av_speed_pu = 10 # define average speed in km/h
    average_TT_pu = [] # average intrazonal tt
    for i in intrazone_dist_list:
        average_TT_pu.append((i / 1000) / (av_speed_pu / 60)) # result in minutes
    np.fill_diagonal(cij_pu, average_TT_pu) # save average TT list

    ## Bikes
    cij_b_df = pd.read_csv(inputs["Cij_bike"], header=None)

    # Convert the dataframe to numpy matrix:
    cij_b = cij_b_df.to_numpy()
    cij_b[cij_b < 0] = 120  # upper limit is 2h, -1 values need to be changed -> DO IN PREPROCESSING
    cij_b[cij_b < 1] = 1  # no link should be below 1

    # Change values in the main diagonal from 0 to intrazone impedance
    av_speed_b = 18  # (Bernardi, 2015)
    average_TT_b = []  # average intrazonal TT
    for i in intrazone_dist_list:
        average_TT_b.append((i / 1000) / (av_speed_b / 60))  # result in minutes
    np.fill_diagonal(cij_b, average_TT_b)  # save average TT as main diag

    # Print the dim of the matrix to check
    logger.warning('cij bike shape:' + str(cij_b.shape))
    cij_b[cij_b < 1] = 1  # repeat of above to remove zeros on diag

    beta = pd.read_csv(inputs['ModeBetas'])

    # Calculated using subzones - update using MTZ when PT times available
    Beta = [beta.loc[f'Car_TT', 'Overall'], beta.loc[f'Bike_TT', 'Overall'], beta.loc[f'Public Transit_TT', 'Overall']]

    print("Using Betas: ", Beta[0], Beta[1], Beta[2])
    ## Do i need this??

    '''
            # Import OD data
            # Import the csv OD private as Pandas DataFrame:
            # OD_pr_df = pd.read_csv(ODPrFilename_csv, header=None)
            # Convert the dataframe to a numpy matrix:
            # OD_pr = OD_pr_df.to_numpy()
            # print('OD private shape: ', OD_pr.shape)

            # Import the csv OD public as Pandas DataFrame:
            # OD_pu_df = pd.read_csv(ODPuFilename_csv, header=None)
            # Convert the dataframe to a numpy matrix:
            # OD_pu = OD_pu_df.to_numpy()
            # print('OD public shape: ', OD_pu.shape)
    '''

    # run relevant models to produce the outputs
    runSGScenarios(cij_pr, cij_pu, cij_b, Beta,zonecodes_SG_list, inputs, outputs, logger)

    # Population maps:
    population_map_creation(inputs, outputs, logger)

    # Flows maps:
    ## long run time!

    ## TODO: rename to correct scenarios

    create_flow_maps = False ## turn on or off here
    if create_flow_maps:
        print("Creating flow maps, might take a while.")
        # flows_output_keys = ["JobsTijPublic2019", "JobsTijPrivate2019", "JobsTijPublic2030", "JobsTijPrivate2030", "JobsTijPublic2045", "JobsTijPrivate2045"]
        flows_output_keys = ["JobsTijPublicBase", "JobsTijPrivateBase",
                             'JobsTijPublicUSnEBC','JobsTijPrivateUSnEBC',
                             'JobsTijPublicUSEBC','JobsTijPrivateUSEBC',
                             'JobsTijPublicUDnEBC','JobsTijPrivateUDnEBC',
                             'JobsTijPublicUDEBC','JobsTijPrivateUDEBC']
        flows_map_creation(inputs, outputs, flows_output_keys)

####################################################################################

def runSGScenarios(cij_pr, cij_pr_ebc, cij_pu, cij_b,Beta, zonecodes_SG_list, inputs, outputs, logger):
    # First run base model to calibrate it with observed trips
    # Run Journey to work model
    DjPred_JtW_2022 = runJourneyToWorkModel(cij_pr, cij_pu, cij_b, Beta, zonecodes_SG_list, inputs, outputs, logger, Scenario="Base")

#####################################################################################
# Journey To Work Model
#####################################################################################

## TODO: define scenario names
## TODO: include bikes

def runJourneyToWorkModel(cij_pr, cij_pu, cij_b, Beta, zonecodes_SG_list, inputs, outputs, logger, Scenario = "Base", Beta_calibrated = None):
    logger.warning("Running Journey To Work " + str(Scenario) + " model.")
    start = time.perf_counter()
    # Singly constrained model:
    # Conserve number of jobs and predict working pop residing in SG zones
    # journeys to work generated by jobs
    # Origins: workplaces
    # Destinations: Zones' households
    # Attractor: floorspace of housing

    """
                        Journey to work       |   Retail model
         Origins:       workplaces            |   households
         Destinations:  households            |   supermarkets
         conserved:     jobs                  |   income
         predicted:     population of zones   |   expenditure @ supermarkets
         attractor:     HH floorspace density |   supermarket floorspace
    """
    if EBikeCity:
        cij_pr = cij_pr_ebc

    ### set up exploratory modeling
    ## define densification thresholds
    housing_threshold = random.random()
    jobs_threshold = random.random()


    print("Housing Threshold set at: ", round(housing_threshold,2))
    print("Jobs Threshold set at: ", round(jobs_threshold,2))


    ## take variable growth
    ## baseline for housing growth is + 400'000 (27%), for jobs growth + 120'000 (20%)
    housing_growth_factor = random.uniform(a = -0.2, b = 0.2) ## +- 20%
    jobs_growth_factor = random.uniform(a = -0.2, b = 0.2) ## +- 20%

    dfcoord = pd.read_csv(inputs["ZoneCoordinates"], usecols= ['zone', 'E_KOORD','N_KOORD'], index_col='zone')
    # load jobs for residential zones
    dfEi = pd.read_csv(inputs['DataEmploymentSG'], usecols=['zone', 'employment'], index_col='zone')
    dfEi.astype({'employment': 'int64'})

    # df_pop = pd.read_csv(inputs['PopulationData'], usecols=['zone','pop_base'], index_col='zone')

    df_floorspace = pd.read_csv(inputs["HhFloorspace"], usecols= ['zone', 'hh_floorspace_density'], index_col= 'zone')
    # Need to sub 0 values in floorspace dataframe with very low values to avoid div by 0
    df_floorspace.replace(0,1 , inplace=True)
    df_floorspace['zone_code'] = df_floorspace.index

    ## include urbanity measure
    df_urbanity = pd.read_csv(inputs["Urbanity"], usecols=['zone', 'urbanity'])

    housing_zones_triggered = df_urbanity[df_urbanity['urbanity'] >= housing_threshold]['zone'].values ## zones to be densified
    jobs_zones_triggered = df_urbanity[df_urbanity['urbanity'] >= jobs_threshold]['zone'].values ## zones with job growth



    """
    For the exploratory modeling part, i only need to run one scenario but keep all outputs.
    
    Set up the run here, but dont change anything as of now in the way the model runs.
    """

    # if we are running a scenario, update the Zones with the new attractors and the numbers of jobs
    if Scenario == 'Base':
        # base scenario, don't change anything
        pass

    elif Scenario == "Exploratory Modeling":
        # Run for exploratory modeling
        dfEi_EM = add_New_Jobs(dfEi, jobs_zones_triggered, jobs_growth_factor)  ## jobs for exploratory modeling (EM)

        df_floorspace_EM = add_New_Housing(df_floorspace, housing_zones_triggered,
                                           housing_growth_factor)  ## floorspace for exploratory modeling (EM)

    elif Scenario == 'UrbanSprawlNoEBC':
        # Urban Sprawl with current cost matrix
        dfEi_USnEBC = pd.read_csv(inputs["DataEmploymentUrbanSprawl"], usecols=['zone', 'employment_UrbanSprawl'], index_col= 'zone')
        dfEi_USnEBC.astype({'employment_UrbanSprawl': 'int64'})

        df_floorspace = pd.read_csv(inputs["HhFloorspaceUrbanSprawl"], usecols=['zone', 'hh_floorspace_UrbanSprawl'], index_col='zone')
        df_floorspace.replace(0,1, inplace=True)

        # df_pop = pd.read_csv(inputs['PopulationData'], usecols=['zone', 'pop_UrbanSprawl'], index_col= 'zone')

    if Scenario == 'Base':
        # Load observed data for calibration
        OD_Pu , OD_Pr = QUANTLHModel.loadODData(inputs)

        # use cij as cost matrix
        m, n = cij_pu.shape
        model = QUANTLHModel(m,n)


        # Cbars are the mean trip length in minutes
        # OBS Cbar public is 47 mins in athens
        # OBS Cbar private is 37 min in ath
        # model.setObsCbar(cbar_public, cbar_private)
        ## TODO: add bikes
        model.setObsCbar(50.64697, 20.47476, 21.30519) ## taken from MZMV

        model.setODMatrix(inputs['ODPublic'], inputs['ODPrivate'])
        model.setAttractorsAj(df_floorspace, 'zone_code','hh_floorspace_density')
        model.setPopulationEi(dfEi, 'zone', 'employment')
        model.setCostMatrixCij(cij_pu, cij_pr, cij_b)

        # Tij, scaling_factors, cbar_k_pred, cbar_k_obs = model.runTwoModes(logger) ## TODO: define runThreeModes

        ## run 3 modes with given beta, calculate Sij

        Tij, beta_k, cbar_k_pred, cbar_k_obs = model.runThreeModes(logger)


        # Save output matrices
        logger.warning("Saving output matrices")
        # Jobs accessibility
        # Job accessibility is the distribution of population around a job location.
        # It’s just the sum of all the population around a job zone divided by the travel time squared.
        # This is scaled so that the total of all i zones comes to 100.

        DjPred_pu = Tij[0].sum(axis = 1)
        Ji_pu = Calculate_Job_Accessibility(DjPred_pu, cij_pu)

        DjPred_pr = Tij[1].sum(axis = 1)
        Ji_pr = Calculate_Job_Accessibility(DjPred_pr, cij_pr)

        DjPred_b = Tij[2].sum(axis = 1)
        Ji_b = Calculate_Job_Accessibility(DjPred_b, cij_b)

        # Save output
        Jobs_accessibility_df = pd.DataFrame({'zone' : zonecodes_SG_list, 'JobsApuBase': Ji_pu, 'JobsAprBase': Ji_pr, 'JobsAbBase': Ji_b})
        Jobs_accessibility_df.to_csv(outputs["JobsAccessibilityBase"])

        # Housing Accessibility
        # Housing accessibility is the distribution of jobs around a housing location.
        # It’s just the sum of all the jobs around a zone divided by the travel time squared.
        # This is scaled so that the total of all i zones comes to 100.

        OiPred_pu = Tij[0].sum(axis = 0)
        Hi_pu = Calculate_Housing_Accessibility(OiPred_pu, cij_pu)

        OiPred_pr = Tij[1].sum(axis = 0)
        Hi_pr = Calculate_Housing_Accessibility(OiPred_pr, cij_pr)

        Oi_pred_b = Tij[2].sum(axis = 0)
        Hi_b = Calculate_Housing_Accessibility(Oi_pred_b, cij_b)

        # Save output
        Housing_accessibility_df = pd.DataFrame({'zone': zonecodes_SG_list, 'HApuBase': Hi_pu, 'HAprBase': Hi_pr, 'HAbBase': Hi_b})
        Housing_accessibility_df.to_csv(outputs["HousingAccessibilityBase"])

        denominator = Tij[0] + Tij[1] + Tij[2]

        denominator = np.where(denominator == 0, 0.01, denominator)
        ms_pu = Tij[0] / denominator
        ms_pr = Tij[1] / denominator
        ms_b = Tij[2] / denominator

        ms_pu_base = np.mean(ms_pu)
        ms_pr_base = np.mean(ms_pr)
        ms_b_base = np.mean(ms_b)

        Acc_total_base = np.sum(ms_pu * (Hi_pu + Ji_pu) + ms_pr * (Hi_pr + Ji_pr) + ms_b * (Hi_b + Ji_b))

        # NOTE: saved to csv, but not easy to read
        # Oj Dj table
        # DfEi is employment
        dfEi['DjPred_pu'] = Tij[0].sum(axis = 1)
        dfEi['DjPred_pr'] = Tij[1].sum(axis = 1)
        dfEi['DjPred_b'] = Tij[2].sum(axis = 1)
        dfEi['DjPred'] = Tij[0].sum(axis = 1) + Tij[1].sum(axis = 1) + Tij[2].sum(axis = 1)
        dfEi['OiPred_pu'] = Tij[0].sum(axis=0)
        dfEi['OiPred_pr'] = Tij[1].sum(axis=0)
        dfEi['OiPred_b'] = Tij[2].sum(axis = 0)
        dfEi['OiPred_Base'] = Tij[0].sum(axis=0) + Tij[1].sum(axis=0) + Tij[2].sum(axis = 0)
        dfEi['Job_accessibility_pu'] = Ji_pu # try calling directly from array
        dfEi['Job_accessibility_pr'] = Ji_pr # Jobs_accessibility_df['HApr22']
        dfEi['Job_accessibility_b'] = Ji_b
        dfEi['Housing_accessibility_pu'] = Hi_pu
        dfEi['Housing_accessibility_pr'] = Hi_pr
        dfEi['Housing_accessibility_b'] = Hi_b
        dfEi['Latitude'] = dfcoord['E_KOORD']
        dfEi['Longitude'] = dfcoord['N_KOORD']
        dfEi.to_csv(outputs["EjOiBase"])

        # Compute the probability of a flow from a zone to any other of the possible point zones
        jobs_probTij = model.computeProbabilities3modes(Tij)

        # Probabilities:
        np.savetxt(outputs['JobsProbTijPublicBase'], jobs_probTij[0], delimiter= ",")
        np.savetxt(outputs['JobsProbTijPrivateBase'], jobs_probTij[1], delimiter = ",")
        np.savetxt(outputs['JobsProbTijBikeBase'], jobs_probTij[2], delimiter=',')

        # People flows
        np.savetxt(outputs['JobsTijPublicBase'], Tij[0], delimiter=",")
        np.savetxt(outputs['JobsTijPrivateBase'], Tij[1], delimiter=",")
        np.savetxt(outputs['JobsTijBikeBase'], Tij[2], delimiter=",")

        if Flows:
            # Geojson flows files - arrows
            flow_zonecodes = pd.read_csv(inputs["ZoneCoordinates"])
            flow_pu = flowArrowsGeoJSON(Tij[0], flow_zonecodes)

            with open(outputs['ArrowsFlowsPublicBase'], 'w') as f:
                dump(flow_pu, f)
            flow_pr = flowArrowsGeoJSON(Tij[1], flow_zonecodes)
            with open(outputs['ArrowsFlowsPrivateBase'], 'w') as f:
                dump(flow_pr, f)
            flow_b = flowArrowsGeoJSON(Tij[2], flow_zonecodes)
            with open(outputs['ArrowsFlowsBikeBase'], 'w') as f:
                dump(flow_b, f)


        # logger.warning("JtW model" + str(Scenario) + "observed cbar [public, private] = " + str(cbar_k_obs))
        # logger.warning("JtW model" + str(Scenario) + "predicted cbar [public, private] = " + str(cbar_k_pred))
        # logger.warning("JtW model" + str(Scenario) + "beta [public, private] = " + str(beta_k))

        # Calculate predicted population

        DjPred = np.zeros(n)
        for k in range(len(Tij)):
            DjPred += Tij[k].sum(axis = 1)
        # Create a dataframe with Zone and people count
        DjPred = pd.DataFrame(DjPred, columns=['population'])
        DjPred['zone'] = zonecodes_SG_list

        end = time.perf_counter()
        logger.warning("Journey to work model run elapsed time (minutes) = " + str((end-start)/60))

        return  DjPred

    elif Scenario == "Exploratory Modeling":
        m, n = cij_pu.shape
        model = QUANTLHModel(m,n)
        model.setAttractorsAj(df_floorspace_EM, 'zone_code', 'hh_floorspace_density')
        model.setPopulationEi(dfEi_EM, 'zone_code', 'employment')
		model.setCostMatrixCij(cij_pu, cij_pr, cij_b)

        Tij, cbar_k = model.run3modes_NoCalibration(Beta)

        jobs_probTij = model.computeProbabilities3modes(Tij)

        # Save output matrices
        logger.warning("Saving output matrices...")

        # Jobs accessibility:
        # Job accessibility is the distribution of population around a job location.
        # It’s just the sum of all the population around a job zone divided by the travel time squared.
        # This is scaled so that the total of all i zones comes to 100.
        DjPred_pu = Tij[0].sum(axis=1)
        Ji_pu = Calculate_Job_Accessibility(DjPred_pu, cij_pu)

        DjPred_pr = Tij[1].sum(axis=1)

		Ji_pr = Calculate_Job_Accessibility(DjPred_pr, cij_pr)

        DjPred_b = Tij[2].sum(axis = 1)
        Ji_b = Calculate_Job_Accessibility(DjPred_b, cij_b)


        # Save output:

        Jobs_accessibility_df = pd.DataFrame(
            {'zone': zonecodes_SG_list, 'JobsApuEM': Ji_pu, 'JobsAprEM': Ji_pr, 'JobsAbEM': Ji_b})
        Jobs_accessibility_df.to_csv(outputs["JobsAccessibilityEM"]) ## TODO: change name for iteration

        # Housing Accessibility
        # Housing accessibility is the distribution of jobs around a housing location.
        # It’s just the sum of all the jobs around a zone divided by the travel time squared.
        # This is scaled so that the total of all i zones comes to 100.

        OiPred_pu = Tij[0].sum(axis=0)
        Hi_pu = Calculate_Housing_Accessibility(OiPred_pu, cij_pu)

        OiPred_pr = Tij[1].sum(axis=0)
		Hi_pr = Calculate_Housing_Accessibility(OiPred_pr, cij_pr)

        OiPred_b = Tij[2].sum(axis = 0)
        Hi_b = Calculate_Housing_Accessibility(OiPred_b, cij_b)


        # Save output:

        Housing_accessibility_df = pd.DataFrame({'zone': zonecodes_SG_list, 'HApuEM': Hi_pu, 'HAprEM': Hi_pr, 'HAbEM': Hi_b})
        Housing_accessibility_df.to_csv(outputs["HousingAccessibilityEM"]) ## TODO: change name for iteration

        ## Calculate mode share matrix wide

        denominator = Tij[0] + Tij[1]+ Tij[2]

        denominator = np.where(denominator == 0, 0.01, denominator)
        ms_pu = Tij[0] / denominator
        ms_pr = Tij[1] / denominator
        ms_b = Tij[2] / denominator


        ms_pu = np.mean(ms_pu)
        ms_pr = np.mean(ms_pr)
        ms_b = np.mean(ms_b)

        Acc_total = np.sum(ms_pu * (Hi_pu + Ji_pu) + ms_pr * (Hi_pr + Ji_pr) + ms_b * (Hi_b +Ji_b))
        # now an Oj Dj table
        # DfEi is employment - really hope these come out in the right order

        dfEi['DjPred_pu'] = Tij[0].sum(axis=1)
        dfEi['DjPred_pr'] = Tij[1].sum(axis=1)
        dfEi['DjPred_b'] = Tij[2].sum(axis = 1)
        dfEi['DjPred'] = Tij[0].sum(axis=1) + Tij[1].sum(axis=1) + Tij[2].sum(axis = 1)
        dfEi['OiPred_pu'] = Tij[0].sum(axis=0)
        dfEi['OiPred_pr'] = Tij[1].sum(axis=0)
        dfEi['OiPred_b'] = Tij[2].sum(axis = 0)
        dfEi['OiPred_EM'] = Tij[0].sum(axis=0) + Tij[1].sum(axis=0) + Tij[2].sum(axis = 0)
        # those need to be called directly from array, otherwise it cannot write a value
        dfEi['Job_accessibility_pu'] = Ji_pu
        dfEi['Job_accessibility_pr'] = Ji_pr
        dfEi['Job_accessibility_b'] = Ji_b
        dfEi['Housing_accessibility_pu'] = Hi_pu
        dfEi['Housing_accessibility_pr'] = Hi_pr
        dfEi['Housing_accessibility_b'] = Hi_b
        dfEi['Latitude'] = dfcoord['E_KOORD']
        dfEi['Longitude'] = dfcoord['N_KOORD']
        dfEi.to_csv(outputs["EjOiEM"])

        # Probabilities:

        np.savetxt(outputs["JobsProbTijPublicEM"], Tij[0], delimiter=",")
        np.savetxt(outputs["JobsProbTijPrivateEM"], Tij[1], delimiter=",")
        np.savetxt(outputs["JobsProbTijBikeEM"], Tij[2], delimiter=',')

        # People flows

        np.savetxt(outputs["JobsTijPublicEM"], Tij[0], delimiter=",")
        np.savetxt(outputs["JobsTijPrivateEM"], Tij[1], delimiter=",")
        np.savetxt(outputs["JobsTijBikeEM"], Tij[2], delimiter=',')

        # Geojson flows files - arrows
        # I need my own zone codes file containing the zonei and greek grid indexes as
        # ZoneCodes_ATH does not contain the information


        if Flows: ## depending if I want the flows as arrow
            flow_zonecodes = pd.read_csv(inputs["ZoneCoordinates"])
            flow_pu = flowArrowsGeoJSON(Tij[0], flow_zonecodes)
            with open(outputs["ArrowsFlowsPublicEM"], 'w') as f:
                dump(flow_pu, f)
            flow_pr = flowArrowsGeoJSON(Tij[1], flow_zonecodes)
            with open(outputs["ArrowsFlowsPrivateEM"], 'w') as f:
                dump(flow_pr, f)
            flow_b = flowArrowsGeoJSON(Tij[2], flow_zonecodes)
            with open(outputs["ArrowsFlowsBikeEM"], 'w') as f:
                dump(flow_b, f)

        logger.warning("JtW model" + str(Scenario) + " cbar [public, private] = " + str(cbar_k))

        DjPred = np.zeros(n)
        for k in range(len(Tij)):
            DjPred += Tij[k].sum(axis=1)
        # Create a dataframe with Zone and people count
        DjPred = pd.DataFrame(DjPred, columns=['population'])
        DjPred['zone'] = zonecodes_SG_list

        end = time.perf_counter()
        # print("Journey to work model run elapsed time (secs) =", end - start)
        logger.warning("Journey to work model run elapsed time (minutes) =" + str((end - start) / 60))

        return DjPred


    elif Scenario == 'UrbanSprawlNoEBC':
        # Use cij as cost matrix
        m, n = cij_pu.shape
        model = QUANTLHModel(m, n)
        model.setAttractorsAj(df_floorspace, 'zone', 'hh_floorspace_UrbanSprawl')
        model.setPopulationEi(dfEi_USnEBC, 'zone', 'employment_UrbanSprawl')
        model.setCostMatrixCij(cij_pu, cij_pr, cij_b)

        Tij, cbar_k = model.run3modes_NoCalibration(Beta_calibrated)
        # Compute the probability of a flow from an MSOA zone to any (i.e. all) of the possible point zones.
        # jobs_probTij = model.computeProbabilities2modes(Tij)
        jobs_probTij = model.computeProbabilities3modes(Tij)

        # Save output matrices
        logger.warning("Saving output matrices...")

        # Jobs accessibility:
        # Job accessibility is the distribution of population around a job location.
        # It’s just the sum of all the population around a job zone divided by the travel time squared.
        # This is scaled so that the total of all i zones comes to 100.
        DjPred_pu = Tij[0].sum(axis=1)
        Ji_pu = Calculate_Job_Accessibility(DjPred_pu, cij_pu)

        DjPred_pr = Tij[1].sum(axis=1)
        Ji_pr = Calculate_Job_Accessibility(DjPred_pr, cij_pr)

        DjPred_b = Tij[2].sum(axis=1)
        Ji_b = Calculate_Job_Accessibility(DjPred_b, cij_b)

        # Save output:
        Jobs_accessibility_df = pd.DataFrame({'zone': zonecodes_SG_list, 'JobsApuUSnEBC': Ji_pu, 'JobsAprUSnEBC': Ji_pr, 'JobsAbUSnEBC': Ji_b})
        Jobs_accessibility_df.to_csv(outputs["JobsAccessibilityUSnEBC"])

        # Housing Accessibility:
        # Housing accessibility is the distribution of jobs around a housing location.
        # It’s just the sum of all the jobs around a zone divided by the travel time squared.

        OiPred_pu = Tij[0].sum(axis=0)
        Hi_pu = Calculate_Housing_Accessibility(OiPred_pu, cij_pu)

        OiPred_pr = Tij[1].sum(axis=0)
        Hi_pr = Calculate_Housing_Accessibility(OiPred_pr, cij_pr)

        OiPred_b = Tij[2].sum(axis = 0)
        Hi_b = Calculate_Housing_Accessibility(OiPred_b, cij_b)

        # Save output:

        Housing_accessibility_df = pd.DataFrame({'zone': zonecodes_SG_list, 'HApuUSnEBC': Hi_pu, 'HAprUSnEBC': Hi_pr, 'HAbUSnEBC': Hi_b})
        Housing_accessibility_df.to_csv(outputs["HousingAccessibilityUSnEBC"])

        # NOTE: these are saved later as csvs, but not with an easy to read formatter
        # np.savetxt("debug_Tij_public_2030.txt", Tij[0], delimiter=",", fmt="%i")
        # np.savetxt("debug_Tij_private_2030.txt", Tij[1], delimiter=",", fmt="%i")
        # now an Oj Dj table
        # DfEi is employment - really hope these come out in the right order

        dfEi['DjPred_pu'] = Tij[0].sum(axis=1)
        dfEi['DjPred_pr'] = Tij[1].sum(axis=1)
        dfEi['DjPred_b'] = Tij[2].sum(axis = 1)
        dfEi['DjPred'] = Tij[0].sum(axis=1) + Tij[1].sum(axis=1) + Tij[2].sum(axis = 1)
        dfEi['OiPred_pu'] = Tij[0].sum(axis=0)
        dfEi['OiPred_pr'] = Tij[1].sum(axis=0)
        dfEi['OiPred_b'] = Tij[2].sum(axis  = 0)
        dfEi['OiPred_USnEBC'] = Tij[0].sum(axis=0) + Tij[1].sum(axis=0) + Tij[2].sum(axis  = 0)
        # those need to be called directly from array, otherwise it cannot write a value
        dfEi['Job_accessibility_pu'] = Ji_pu
        dfEi['Job_accessibility_pr'] = Ji_pr
        dfEi['Job_accessibility_b'] = Ji_b
        dfEi['Housing_accessibility_pu'] = Hi_pu
        dfEi['Housing_accessibility_pr'] = Hi_pr
        dfEi['Housing_accessibility_b'] = Hi_b
        dfEi['Latitude'] = dfcoord['E_KOORD']
        dfEi['Longitude'] = dfcoord['N_KOORD']
        dfEi.to_csv(outputs["EjOiUSnEBC"])

        # Probabilities:
        np.savetxt(outputs["JobsProbTijPublicUSnEBC"], Tij[0], delimiter=",")
        np.savetxt(outputs["JobsProbTijPrivateUSnEBC"], Tij[1], delimiter=",")
        np.savetxt(outputs["JobsProbTijBikeUSnEBC"], Tij[1], delimiter=",")

        # People flows
        np.savetxt(outputs["JobsTijPublicUSnEBC"], Tij[0], delimiter=",")
        np.savetxt(outputs["JobsTijPrivateUSnEBC"], Tij[1], delimiter=",")
        np.savetxt(outputs["JobsTijBikeUSnEBC"], Tij[1], delimiter=",")

        # Geojson flows files - arrows
        # I need my own zone codes file containing the zonei and greek grid indexes as
        # ZoneCodes_ATH does not contain the information
        if Flows:
            flow_zonecodes = pd.read_csv(inputs["ZoneCoordinates"])
            flow_pu = flowArrowsGeoJSON(Tij[0], flow_zonecodes)
            with open(outputs["ArrowsFlowsPublicUSnEBC"], 'w') as f:
                dump(flow_pu, f)
            flow_pr = flowArrowsGeoJSON(Tij[1], flow_zonecodes)
            with open(outputs["ArrowsFlowsPrivateUSnEBC"], 'w') as f:
                dump(flow_pr, f)
            flow_b = flowArrowsGeoJSON(Tij[2], flow_zonecodes)
            with open(outputs["ArrowsFlowsBikeUSnEBC"], 'w') as f:
                dump(flow_b, f)

        logger.warning("JtW model" + str(Scenario) + " cbar [public, private] = " + str(cbar_k))

        DjPred = np.zeros(n)
        for k in range(len(Tij)):
            DjPred += Tij[k].sum(axis=1)
        # Create a dataframe with Zone and people count
        DjPred = pd.DataFrame(DjPred, columns=['population'])
        DjPred['zone'] = zonecodes_SG_list

        end = time.perf_counter()
        # print("Journey to work model run elapsed time (secs) =", end - start)
        logger.warning("Journey to work model run elapsed time (minutes) =" + str((end - start)/60))

        return DjPred
#################################################################################

def Calculate_Job_Accessibility(DjPred, cij):

    # Job accessibility is the distribution of population around a job location.
    # It’s just the sum of all the population around a job zone divided by the travel time squared.
    # This is scaled so that the total of all i zones comes to 100.

    Ji = np.zeros(len(DjPred))
    for i in range(len(Ji)):
        for j in range(len(Ji)):
            Ji[i] += DjPred[j] / (cij[i,j] * cij[i,j]) # DjPred is total residential

    ## remove the scaling to get comparable numbers between modes
    # # now scale to 100
    # Sum = 0
    # for i in range(len(Ji)): Sum += Ji[i]
    # for i in range(len(Ji)): Ji[i] = 100.0 * Ji[i] / Sum
    return Ji

#############################################################################


def Calculate_Housing_Accessibility(OiPred, cij):
    # Housing accessibility is the distribution of population around a job location.
    # It’s just the sum of all the jobs around a zone divided by the travel time squared.
    # This is scaled so that the total of all i zones comes to 100.

    Hi = np.zeros(len(OiPred))

    # Calculate housing accessibility for public transport
    for i in range(len(Hi)):
        for j in range(len(Hi)):
            Hi[i] += OiPred[j] / (cij[i,j] * cij[i,j]) # OiPred_pu is employment totals

    ## remove the scaling to get comparable numbers between modes
    # # now scale to 100
    # Sum = 0
    # for i in range(len(Hi)): Sum += Hi[i]
    # for i in range(len(Hi)): Hi[i] = 100.0 * Hi[i] / Sum
    return Hi

#################################################################

def add_New_Housing(df, triggered_array, growth_factor):
    ## calculate new floorspace to be distributed
    floorspace_to_distribute = 400000 + (growth_factor* 400000) ### 400'000 is the baseline +27% as projected by the canton

    ## calculate share of active zones
    active_zones = df[df.index.isin(triggered_array)] ## get zones that are triggered by threshold
    active_zones['share'] = active_zones['hh_floorspace_density'] / np.sum(active_zones['hh_floorspace_density']) ## calculate their share
    active_zones['floorspace_new'] = active_zones['hh_floorspace_density'] + (active_zones['share'] * floorspace_to_distribute) ## calculate new floorspace based on share

    ## add back to df -> creates mismatch
    df = pd.merge(df, active_zones, left_index=True, right_index=True, how="left")
    # write into same column
    df['hh_floorspace_density'] = df['floorspace_new'].where(df.index.isin(triggered_array), df['hh_floorspace_density_x'])
    # drop unnecessary cols
    df = df.drop(labels=['hh_floorspace_density_x', 'hh_floorspace_density_y', 'share', 'floorspace_new','zone_code_x','zone_code_y'], axis=1)
    # add index as col zone
    df['zone_code'] = df.index

    return df

#################################################################

def add_New_Jobs(df, triggered_array, growth_factor):
    ## calculate new jobs to distribute
    jobs_to_distribute = 120000 +(growth_factor * 120000) ### 120'000 is the projected growth in jobs by 2050 (Canton of Zurich, 2024)

    ## calculate share of active zones
    active_zones = df[df.index.isin(triggered_array)] ## get active zones
    active_zones['share'] = active_zones['employment'] / np.sum(active_zones['employment']) ## get share per zone
    active_zones['employment_new'] = active_zones['employment'] + (active_zones['share'] * jobs_to_distribute) ## calculate new jobs based on share

    df = pd.merge(df, active_zones, left_index=True, right_index=True, how="left")
    df['employment'] = df['employment_new'].where(df.index.isin(triggered_array), df['employment_x']) ## add new employment if a
    df = df.drop(['employment_x', 'employment_y', 'share','employment_new'], axis = 1)
    df['zone_code'] = df.index
    return df

def add_New_Housing_exploratory(df, triggered_array, growth_factor):
    ## calculate new floorspace to be distributed
    floorspace_to_distribute = np.sum(df['hh_floorspace_density']) * growth_factor ### 400'000 is the baseline +27% as projected by the canton

    ## calculate share of active zones
    active_zones = df[df.index.isin(triggered_array)] ## get zones that are triggered by threshold
    active_zones['share'] = active_zones['hh_floorspace_density'] / np.sum(active_zones['hh_floorspace_density']) ## calculate their share
    active_zones['floorspace_new'] = active_zones['hh_floorspace_density'] + (active_zones['share'] * floorspace_to_distribute) ## calculate new floorspace based on share

    ## add back to df -> creates mismatch
    df = pd.merge(df, active_zones, left_index=True, right_index=True, how="left")
    # write into same column
    df['hh_floorspace_density'] = df['floorspace_new'].where(df.index.isin(triggered_array), df['hh_floorspace_density_x'])
    # drop unnecessary cols
    df = df.drop(labels=['hh_floorspace_density_x', 'hh_floorspace_density_y', 'share', 'floorspace_new','zone_code_x','zone_code_y'], axis=1)
    # add index as col zone
    df['zone_code'] = df.index

    return df


def add_new_Jobs_exploratory(df, triggered_array, growth_factor):
    ## calculate new jobs to distribute
    jobs_to_distribute = np.sum(df['employment']) * growth_factor ### 120'000 is the projected growth in jobs by 2050 (Canton of Zurich, 2024)

    ## calculate share of active zones
    active_zones = df[df.index.isin(triggered_array)] ## get active zones
    active_zones['share'] = active_zones['employment'] / np.sum(active_zones['employment']) ## get share per zone
    active_zones['employment_new'] = active_zones['employment'] + (active_zones['share'] * jobs_to_distribute) ## calculate new jobs based on share

    df = pd.merge(df, active_zones, left_index=True, right_index=True, how="left")
    df['employment'] = df['employment_new'].where(df.index.isin(triggered_array), df['employment_x']) ## add new employment if a
    df = df.drop(['employment_x', 'employment_y', 'share','employment_new'], axis = 1)
    df['zone_code'] = df.index
    return df
### Define own function for exploratory modeling
### Essentially the same function as runJourneyToWorkModel but with adapted inputs

def calcLUTIBaseline(cij_pu, cij_pr, cij_b,  inputs, zonecodes_SG_list, # Input Data
                     growth_factor_housing, growth_factor_jobs,
                 housing_threshold, jobs_threshold):

    ### Read spatial data
    dfcoord = pd.read_csv(inputs["ZoneCoordinates"], usecols=['zone', 'E_KOORD', 'N_KOORD'], index_col='zone')
    # load jobs for residential zones
    dfEi = pd.read_csv(inputs['DataEmploymentSG'], usecols=['zone', 'employment'], index_col='zone')
    dfEi.astype({'employment': 'int64'})

    # load housing for residential zones
    df_floorspace = pd.read_csv(inputs["HhFloorspace"], usecols=['zone', 'hh_floorspace_density'], index_col='zone')
    # Need to sub 0 values in floorspace dataframe with very low values to avoid div by 0
    df_floorspace.replace(0, 1, inplace=True)
    df_floorspace['zone_code'] = df_floorspace.index

    ## include urbanity measure (0 is urban sprawl, 1 is only the most dense places)
    df_urbanity = pd.read_csv(inputs["Urbanity"], usecols=['zone', 'urbanity'])

    ## zones to be densified
    housing_zones_triggered = df_urbanity[df_urbanity['urbanity'] >= housing_threshold][
        'zone'].values
    ## zones with job growth
    jobs_zones_triggered = df_urbanity[df_urbanity['urbanity'] >= jobs_threshold][
        'zone'].values

    ## add new jobs
    df_Ei_baseline = add_new_Jobs_exploratory(dfEi,jobs_zones_triggered, growth_factor_jobs)

    ## add new housing
    df_floorspace_baseline = add_New_Housing_exploratory(df_floorspace, housing_zones_triggered, growth_factor_housing)


    ### Set up journey to work model - current situation
    ## Calculate baseline values and compare
    m, n = cij_pu.shape
    model = QUANTLHModel(m, n)
    model.setAttractorsAj(df_floorspace_baseline, 'zone_code', 'hh_floorspace_density')  ## load housing
    model.setPopulationEi(df_Ei_baseline, 'zone', 'employment')  ## load employment
    ## set costs
    model.setCostMatrixCij(cij_pu, cij_pr, cij_b)

    Beta = [0.15, 0.3, 0.2]
    ## Calculate flows and travel times per mode
    Tij, cbar_k = model.run3modes_NoCalibration(Beta)

    # Save output matrices

    # Jobs accessibility:
    # Job accessibility is the distribution of population around a job location.
    # It’s just the sum of all the population around a job zone divided by the travel time squared.

    DjPred_pu = Tij[0].sum(axis=1)
    Ji_pu = Calculate_Job_Accessibility(DjPred_pu, cij_pu)

    DjPred_pr = Tij[1].sum(axis=1)
    DjPred_b = Tij[2].sum(axis=1)

    Ji_pr = Calculate_Job_Accessibility(DjPred_pr, cij_pr)
    Ji_b = Calculate_Job_Accessibility(DjPred_b, cij_b)

    # Save output:

    Jobs_accessibility_df = pd.DataFrame(
        {'zone': zonecodes_SG_list, 'JobsApuBase': Ji_pu, 'JobsAprBase': Ji_pr, 'JobsAbBase': Ji_b})
    # Jobs_accessibility_df.to_csv(outputs["JobsAccessibilityEM"])

    # Housing Accessibility
    # Housing accessibility is the distribution of jobs around a housing location.
    # It’s just the sum of all the jobs around a zone divided by the travel time squared.
    # This is scaled so that the total of all i zones comes to 100.

    OiPred_pu = Tij[0].sum(axis=0)
    Hi_pu = Calculate_Housing_Accessibility(OiPred_pu, cij_pu)

    OiPred_pr = Tij[1].sum(axis=0)
    OiPred_b = Tij[2].sum(axis=0)
    Hi_pr = Calculate_Housing_Accessibility(OiPred_pr, cij_pr)
    Hi_b = Calculate_Housing_Accessibility(OiPred_b, cij_b)

    # Save output:

    Housing_accessibility_df = pd.DataFrame(
        {'zone': zonecodes_SG_list, 'HApuBase': Hi_pu, 'HAprBase': Hi_pr, 'HAbBase': Hi_b})
    # Housing_accessibility_df.to_csv(outputs["HousingAccessibilityEM"])

    ## Calculate denominator over all modes
    denominator = Tij[0] + Tij[1] + Tij[2]
    denominator = np.where(denominator == 0, 0.01, denominator)  # catch div by 0

    ## Calculate mode share per OD pair
    ms_pu = Tij[0] / denominator
    ms_pr = Tij[1] / denominator
    ms_b = Tij[2] / denominator

    ## Take mean over all matrices to obtain single value
    ms_pu = np.mean(ms_pu)
    ms_pr = np.mean(ms_pr)
    ms_b = np.mean(ms_b)

    ## The sum of ms_k should be 1, if not, the run should be discarded
    # sum_ms_base = ms_pu + ms_pr + ms_b

    ### Total accessibility weighted by modal split
    Acc_total_base = np.sum(ms_pu * (Hi_pu + Ji_pu) + ms_pr * (Hi_pr + Ji_pr) + ms_b * (Hi_b + Ji_b))

    # now an Oj Dj table
    # DfEi is employment - really hope these come out in the right order

    tt_pu = np.sum(Tij[0] * cij_pu)
    tt_pr = np.sum(Tij[1] * cij_pr)
    tt_b = np.sum(Tij[2] * cij_b)

    ## Accessibility Zurich

    ### Filter rows from final dataframe Housing accessibility and jobs accessibility
    ### Filter rows that start with 2610

    Housing_accessibility_df['zone'] = Housing_accessibility_df['zone'].astype(str)
    Housing_acc_zurich = Housing_accessibility_df[Housing_accessibility_df['zone'].str.contains("2610")]

    Jobs_accessibility_df['zone'] = Jobs_accessibility_df['zone'].astype(str)
    Jobs_acc_zurich = Jobs_accessibility_df[Jobs_accessibility_df['zone'].str.contains("2610")]

    Acc_zurich_base = np.sum(ms_pu * (Housing_acc_zurich['HApuBase'] + Jobs_acc_zurich['JobsApuBase']) +
                             ms_pr * (Housing_acc_zurich['HAprBase'] + Jobs_acc_zurich['JobsAprBase']) +
                             ms_b * (Housing_acc_zurich['HAbBase'] + Jobs_acc_zurich['JobsAbBase']))

    return Acc_total_base, Acc_zurich_base, ms_pu, ms_pr, ms_b, tt_pu, tt_pr, tt_b


def runLUTIModel(cij_pu, cij_pr, cij_pr_ebc, cij_b, cij_b_ebc, inputs, Beta_pu, Beta_pr, Beta_b, zonecodes_SG_list, # Input Data
                 housing_threshold, jobs_threshold, housing_growth_factor, jobs_growth_factor, # Uncertainties
                 EBikeCity, Acc_total_base, Acc_zurich_base,
                 dfEi,df_floorspace, df_urbanity ): # Levers
    # Singly constrained model:
    # Conserve number of jobs and predict working pop residing in SG zones
    # journeys to work generated by jobs
    # Origins: workplaces
    # Destinations: Zones' households
    # Attractor: floorspace of housing

    ## Returns: mode share of PT, Car, Bike, weighted accessibilities, predicted people movement

    """
                        Journey to work       |   Retail model
         Origins:       workplaces            |   households
         Destinations:  households            |   supermarkets
         conserved:     jobs                  |   income
         predicted:     population of zones   |   expenditure @ supermarkets
         attractor:     HH floorspace density |   supermarket floorspace
    """


    ## Adapt Spatial distribution
    housing_threshold = housing_threshold/100
    jobs_threshold = jobs_threshold/100

    ### Read spatial data
    # dfcoord = pd.read_csv(inputs["ZoneCoordinates"], usecols=['zone', 'E_KOORD', 'N_KOORD'], index_col='zone')
    # load jobs for residential zones

    ## zones to be densified
    housing_zones_triggered = df_urbanity[df_urbanity['urbanity'] >= housing_threshold][
        'zone'].values
    ## zones with job growth
    jobs_zones_triggered = df_urbanity[df_urbanity['urbanity'] >= jobs_threshold][
        'zone'].values

    ### Adapt for EM
    dfEi_EM = add_new_Jobs_exploratory(dfEi, jobs_zones_triggered, jobs_growth_factor)  ## jobs for exploratory modeling (EM)

    df_floorspace_EM = add_New_Housing_exploratory(df_floorspace, housing_zones_triggered,
                                       housing_growth_factor)  ## floorspace for exploratory modeling (EM)

    ## Import baseline
    Acc_total_base = Acc_total_base
    Acc_zurich_base = Acc_zurich_base


    ### Set up journey to work model - future situation
    m, n = cij_pu.shape
    model = QUANTLHModel(m, n)
    model.setAttractorsAj(df_floorspace_EM, 'zone_code', 'hh_floorspace_density')  ## load housing
    model.setPopulationEi(dfEi_EM, 'zone_code', 'employment')  ## load employment
    Beta = [Beta_pu, Beta_pr, Beta_b]
    ## set costs
    if EBikeCity == 1:
        model.setCostMatrixCij(cij_pu, cij_pr_ebc, cij_b_ebc)
    if not EBikeCity == 1:
        model.setCostMatrixCij(cij_pu, cij_pr, cij_b)

    ## Calculate flows and travel times per mode
    Tij, cbar_k = model.run3modes_NoCalibration(Beta)

    # jobs_probTij = model.computeProbabilities3modes(Tij)

    # Save output matrices

    #### Jobs accessibility ####
    # Job accessibility is the distribution of population around a job location.
    # It’s just the sum of all the population around a job zone divided by the travel time squared.

    DjPred_pu = Tij[0].sum(axis=1)
    Ji_pu = Calculate_Job_Accessibility(DjPred_pu, cij_pu)

    DjPred_pr = Tij[1].sum(axis=1)
    DjPred_b = Tij[2].sum(axis=1)

    if EBikeCity == 1:
        Ji_pr = Calculate_Job_Accessibility(DjPred_pr, cij_pr_ebc)
        Ji_b = Calculate_Job_Accessibility(DjPred_b, cij_b_ebc)

    if not EBikeCity == 1:
        Ji_pr = Calculate_Job_Accessibility(DjPred_pr, cij_pr)
        Ji_b = Calculate_Job_Accessibility(DjPred_b, cij_b)

    # Save output:

    Jobs_accessibility_df = pd.DataFrame(
        {'zone': zonecodes_SG_list, 'JobsApuEM': Ji_pu, 'JobsAprEM': Ji_pr, 'JobsAbEM': Ji_b})
    # Jobs_accessibility_df.to_csv(outputs["JobsAccessibilityEM"])

    #### Housing Accessibility ####
    # Housing accessibility is the distribution of jobs around a housing location.
    # It’s just the sum of all the jobs around a zone divided by the travel time squared.
    # This is scaled so that the total of all i zones comes to 100.

    OiPred_pu = Tij[0].sum(axis=0)
    Hi_pu = Calculate_Housing_Accessibility(OiPred_pu, cij_pu)

    OiPred_pr = Tij[1].sum(axis=0)
    OiPred_b = Tij[2].sum(axis=0)
    if EBikeCity == 1:
        Hi_pr = Calculate_Housing_Accessibility(OiPred_pr, cij_pr_ebc)
        Hi_b = Calculate_Housing_Accessibility(OiPred_b, cij_b_ebc)
    if not EBikeCity == 1:
        Hi_pr = Calculate_Housing_Accessibility(OiPred_pr, cij_pr)
        Hi_b = Calculate_Housing_Accessibility(OiPred_b, cij_b)

    # Save output:

    Housing_accessibility_df = pd.DataFrame(
        {'zone': zonecodes_SG_list, 'HApuEM': Hi_pu, 'HAprEM': Hi_pr, 'HAbEM': Hi_b})
    # Housing_accessibility_df.to_csv(outputs["HousingAccessibilityEM"])  ## TODO: change name for iteration

    ## Calculate denominator over all modes
    denominator = Tij[0] + Tij[1] + Tij[2]
    denominator = np.where(denominator == 0, 0.01, denominator)  # catch div by 0

    ## Calculate mode share per OD pair
    ms_pu = Tij[0] / denominator
    ms_pr = Tij[1] / denominator
    ms_b = Tij[2] / denominator

    ## Take mean over all matrices to obtain single value
    ms_pu = np.mean(ms_pu)
    ms_pr = np.mean(ms_pr)
    ms_b = np.mean(ms_b)

    ## The sum of ms_k should be 1, if not, the run should be discarded
    sum_ms = ms_pu + ms_pr + ms_b

    ### Total accessibility weighted by modal split
    Acc_total = np.sum(ms_pu * (Hi_pu + Ji_pu) + ms_pr * (Hi_pr + Ji_pr) + ms_b * (Hi_b + Ji_b))

    # now an Oj Dj table
    # DfEi is employment - really hope these come out in the right order

    ## Calculate new distribution of people
    DjPred = np.zeros(n)
    for k in range(len(Tij)):
        DjPred += Tij[k].sum(axis=1)
    # Create a dataframe with Zone and people count
    DjPred = pd.DataFrame(DjPred, columns=['population'])
    DjPred['zone'] = zonecodes_SG_list

    #### Mode Share Zurich ####

    ## Filter trips for zurich rows and cols 902 until 1219 (last one)
    Tij_SG_pu = Tij[0][902:, 902:]
    Tij_SG_pr = Tij[1][902:, 902:]
    Tij_SG_b = Tij[2][902:, 902:]

    ## calculate mode share for zurich
    denominator_SG = Tij_SG_pu + Tij_SG_pr + Tij_SG_b

    ms_SG_pu = Tij_SG_pu / denominator_SG
    ms_SG_pr = Tij_SG_pr / denominator_SG
    ms_SG_b = Tij_SG_b / denominator_SG

    ## Take mean over all matrices to obtain single value
    ms_SG_pu = np.nanmean(ms_SG_pu)
    ms_SG_pr = np.nanmean(ms_SG_pr)
    ms_SG_b = np.nanmean(ms_SG_b)

    #### Accessibility Zurich ####

    ### Filter rows from final dataframe Housing accessibility and jobs accessibility
    ### Filter rows that start with 2610

    # Housing_accessibility_df['zone'] = Housing_accessibility_df['zone'].astype(str)
    Housing_acc_zurich = Housing_accessibility_df[Housing_accessibility_df['zone'].str.contains("2610")]

    Jobs_accessibility_df['zone'] = Jobs_accessibility_df['zone'].astype(str)
    Jobs_acc_zurich = Jobs_accessibility_df[Jobs_accessibility_df['zone'].str.contains("2610")]

    Acc_zurich = np.sum(ms_SG_pu * (Housing_acc_zurich['HApuEM'] + Jobs_acc_zurich['JobsApuEM']) +
                        ms_SG_pr * (Housing_acc_zurich['HAprEM'] + Jobs_acc_zurich['JobsAprEM']) +
                        ms_SG_b * (Housing_acc_zurich['HAbEM'] + Jobs_acc_zurich['JobsAbEM']))

    diff_acc_total = Acc_total - Acc_total_base
    diff_acc_SG = Acc_zurich - Acc_zurich_base

    #### Calculate total travel time ####

    # PT
    tt_pu = np.sum(Tij[0] * cij_pu)

    if EBikeCity:
        # Car
        tt_pr = np.sum(Tij[1] * cij_pr_ebc)
        # Bike
        tt_b = np.sum(Tij[2] * cij_b_ebc)
    if not EBikeCity:
        # Car
        tt_pr = np.sum(Tij[1] * cij_pr)
        # Bike
        tt_b = np.sum(Tij[2] * cij_b)


    return (ms_pu, ms_pr, ms_b,
            ms_SG_pu, ms_SG_pr, ms_SG_b,
            Acc_total, Acc_zurich,
            Acc_total_base, Acc_zurich_base,
            tt_pu, tt_pr, tt_b)
