import pickle
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from shapely.strtree import STRtree
import geopandas as gpd
import pandana as pdna
from tqdm import tqdm
from trip_distribution import *
from joblib import Parallel, delayed


def compile_vehicle_ownership_attributes(road_tt, proximity_bike, proximity_mrt,
                                         proximity_bus, dist2parking,
                                         demographics, zone_codes):

    # Final output is just a list of land-use attributes for each zone
    avg_road_tt = road_tt.groupby("Origin").agg({"TT":"mean"}).reset_index()
    avg_road_tt.columns = ['ZONE_CODE', 'AVG_ROADNT_TT']

    proximity_bike.columns = ['ZONE_CODE', 'AVG_GEOGDIST_KM_PCN']
    proximity_bike['AVG_GEOGDIST_KM_PCN'] = proximity_bike['AVG_GEOGDIST_KM_PCN']/1000

    proximity_mrt.columns = ['ZONE_CODE', 'AVG_WALKDIST_MRT']
    proximity_mrt['AVG_WALKDIST_MRT'] = proximity_mrt['AVG_WALKDIST_MRT']/1000

    proximity_bus.columns = ['ZONE_CODE', 'AVG_WALKDIST_BUS']
    proximity_bus['AVG_WALKDIST_BUS'] = proximity_bus['AVG_WALKDIST_BUS']/1000

    dist2parking.columns = ['ZONE_CODE', 'DIST2RESIPARKING']
    dist2parking['DIST2RESIPARKING'] = dist2parking['DIST2RESIPARKING']/1000

    demographics = demographics[['SUB_MTZ_NO', 'JOB_DENS', 'POP_DENS']]
    demographics.columns = ['ZONE_CODE', 'JOB_DENS', 'POP_DENS']

    dat = pd.DataFrame()
    dat['ZONE_CODE'] = zone_codes

    dat = dat.merge(avg_road_tt, on='ZONE_CODE', how='left'
              ).merge(proximity_bike, on='ZONE_CODE', how='left'
                      ).merge(proximity_mrt, on='ZONE_CODE', how='left'
                              ).merge(proximity_bus, on='ZONE_CODE', how='left'
                                      ).merge(dist2parking, on='ZONE_CODE', how='left'
                                              ).merge(demographics, on='ZONE_CODE', how='left')

    return dat
def bike_ownership_choice(row, veh_coefficients):
    v_bike_ownership = veh_coefficients['bo']['asc_bike'] + \
                       veh_coefficients['bo']['b_elderly'] * row['ELDERLY'] + \
                       veh_coefficients['bo']['b_child'] * row['CHILD'] + \
                       veh_coefficients['bo']['b_max_income'] * row['MAX_INCOME'] + \
                       veh_coefficients['bo']['b_dist2Bus'] * row['AVG_WALKDIST_BUS'] + \
                       veh_coefficients['bo']['b_dist2Mrt'] * row['AVG_WALKDIST_MRT'] + \
                       veh_coefficients['bo']['b_distPCN'] * row['AVG_GEOGDIST_KM_PCN'] + \
                       veh_coefficients['bo']['b_jobDens'] * row['JOB_DENS'] + \
                       veh_coefficients['bo']['b_popDens'] * row['POP_DENS']
    v_neither = 0

    P_bike = np.exp(v_bike_ownership) / (np.exp(v_bike_ownership) + np.exp(v_neither))
    P_none = np.exp(v_neither) / (np.exp(v_bike_ownership) + np.exp(v_neither))

    probabilities = {1: P_bike, 0: P_none}
    choice = max(probabilities, key=probabilities.get)

    return choice
def car_ownership_choice(row, veh_coefficients):
    v_car_ownership = veh_coefficients['co']['asc_car'] + \
                      veh_coefficients['co']['b_elderly'] * row['ELDERLY'] + \
                      veh_coefficients['co']['b_child'] * row['CHILD'] + \
                      veh_coefficients['co']['b_max_income'] * row['MAX_INCOME'] + \
                      veh_coefficients['co']['b_dual_income'] * row['DUAL_INCOME'] + \
                      veh_coefficients['co']['b_high_income'] * row['HIGH_INCOME'] + \
                      veh_coefficients['co']['b_dist2Mrt'] * row['AVG_WALKDIST_MRT'] + \
                      veh_coefficients['co']['b_ttROAD'] * row['AVG_ROADNT_TT'] + \
                      veh_coefficients['co']['b_distParking'] * row['DIST2RESIPARKING'] + \
                      veh_coefficients['co']['b_jobDens'] * row['JOB_DENS'] + \
                      veh_coefficients['co']['b_popDens'] * row['POP_DENS']

    v_neither = 0

    P_car = np.exp(v_car_ownership) / (np.exp(v_car_ownership) + np.exp(v_neither))
    P_none = np.exp(v_neither) / (np.exp(v_car_ownership) + np.exp(v_neither))

    probabilities = {1: P_car, 0:P_none}
    choice = max(probabilities, key=probabilities.get)

    return choice

def calculate_mode_split(dataframe, return_observed=False):
    value_counts = dataframe['MODE_CHOICE'].value_counts()
    co_value_counts = pd.DataFrame(dataframe[dataframe['HH_CAR_OWNERSHIP'] == 1]['MODE_CHOICE'].value_counts()).rename(
        columns={'count': 'CarOwners'})
    nco_value_counts = pd.DataFrame(dataframe[dataframe['HH_CAR_OWNERSHIP'] != 1]['MODE_CHOICE'].value_counts()).rename(
        columns={'count': 'NonCarOwners'})
    combined = pd.concat([pd.DataFrame(co_value_counts), pd.DataFrame(nco_value_counts)], axis=1)
    combined['CAR_MODE_SPLIT'] = combined['CarOwners'] / combined['CarOwners'].sum()
    combined['NONCAR_MODE_SPLIT'] = combined['NonCarOwners'] / combined['NonCarOwners'].sum()

    if return_observed:
        obs_co_value_counts = pd.DataFrame(
            dataframe[dataframe['HH_CAR_OWNERSHIP'] == 1]['TRIPLEG_MODES'].value_counts()).rename(
            columns={'count': 'Obs_CarOwners'})
        obs_nco_value_counts = pd.DataFrame(
            dataframe[dataframe['HH_CAR_OWNERSHIP'] != 1]['TRIPLEG_MODES'].value_counts()).rename(
            columns={'count': 'Obs_NonCarOwners'})
        combined = pd.concat([combined, pd.DataFrame(obs_co_value_counts), pd.DataFrame(obs_nco_value_counts)], axis=1)
        combined['OBS_CAR_MODE_SPLIT'] = combined['Obs_CarOwners'] / combined['OBS_CAR_MODE_SPLIT'].sum()
        combined['OBS_NONCAR_MODE_SPLIT'] = combined['Obs_NonCarOwners'] / combined['OBS_NonCarOwners'].sum()

    return combined

def compile_mode_choice_attributes(roads_time_input, bike_time_input,
                                   pt_ivt_input, pt_ovt_input, pt_modes_input,
                                   walk_time_input, carP2P_dist_input, bikeP2P_dist_input,
                                   parking_costs):
    roads_time = roads_time_input.copy()
    bike_time = bike_time_input.copy()
    pt_ivt = pt_ivt_input.copy()
    pt_ovt = pt_ovt_input.copy()
    carP2P_dist = carP2P_dist_input.copy()
    bikeP2P_dist = bikeP2P_dist_input.copy()
    walk_time = walk_time_input.copy()
    pt_modes = pt_modes_input.copy()

    roads_time = roads_time.melt(ignore_index=False).reset_index()
    roads_time.columns = ['ORIGIN_MTZ_1', 'DEST_MTZ_1', 'CAR_TT']
    roads_time['CAR_TT'] = roads_time['CAR_TT'].clip(upper=90)

    bike_time = bike_time.melt(ignore_index=False).reset_index()
    bike_time.columns = ['ORIGIN_MTZ_1', 'DEST_MTZ_1', 'BIKE_TT']
    bike_time['BIKE_TT'] = bike_time['BIKE_TT'].clip(upper=90)

    pt_ivt = pt_ivt.melt(ignore_index=False).reset_index()
    pt_ivt.columns = ['ORIGIN_MTZ_1', 'DEST_MTZ_1', 'PT_IVT']
    pt_ovt = pt_ovt.melt(ignore_index=False).reset_index()
    pt_ovt.columns = ['ORIGIN_MTZ_1', 'DEST_MTZ_1', 'PT_OVT']
    pt_modes = pt_modes.melt(ignore_index=False).reset_index()
    pt_modes.columns = ['ORIGIN_MTZ_1', 'DEST_MTZ_1', 'PT_NUM_MODES']
    pt_ivt['PT_IVT'] = pt_ivt['PT_IVT'].clip(lower=0.1, upper=90)
    pt_ovt['PT_OVT'] = pt_ovt['PT_OVT'].clip(upper=90)

    walk_time = walk_time.melt(ignore_index=False).reset_index()
    walk_time.columns = ['ORIGIN_MTZ_1', 'DEST_MTZ_1', 'WALK_TT']
    walk_time['WALK_TT'] = walk_time['WALK_TT'].clip(upper=90)

    carP2P_dist = carP2P_dist.melt(ignore_index=False).reset_index()
    carP2P_dist.columns = ['ORIGIN_MTZ_1', 'DEST_MTZ_1', 'CAR_DIST_M']
    carP2P_dist['CAR_DIST_KM'] = carP2P_dist['CAR_DIST_M'] / 1000

    bikeP2P_dist = bikeP2P_dist.melt(ignore_index=False).reset_index()
    bikeP2P_dist.columns = ['ORIGIN_MTZ_1', 'DEST_MTZ_1', 'BIKE_DIST_M']
    bikeP2P_dist['BIKE_DIST_KM'] = bikeP2P_dist['BIKE_DIST_M'] / 1000


    compilation = roads_time.copy()
    compilation = compilation.merge(bike_time, on=['ORIGIN_MTZ_1', 'DEST_MTZ_1'], how='left')
    compilation = compilation.merge(pt_ivt, on=['ORIGIN_MTZ_1', 'DEST_MTZ_1'], how='left')
    compilation = compilation.merge(pt_ovt, on=['ORIGIN_MTZ_1', 'DEST_MTZ_1'], how='left')
    compilation = compilation.merge(pt_modes, on=['ORIGIN_MTZ_1', 'DEST_MTZ_1'], how='left')
    compilation = compilation.merge(walk_time, on=['ORIGIN_MTZ_1', 'DEST_MTZ_1'], how='left')
    compilation = compilation.merge(carP2P_dist, on=['ORIGIN_MTZ_1', 'DEST_MTZ_1'], how='left')
    compilation = compilation.merge(bikeP2P_dist, on=['ORIGIN_MTZ_1', 'DEST_MTZ_1'], how='left')

    compilation['CAR_CAB_FARE'] = np.round(
        ((compilation['CAR_DIST_KM'] - compilation['CAR_DIST_KM'].min()) / (
                compilation['CAR_DIST_KM'].max() - compilation['CAR_DIST_KM'].min())) * (50 - 6) + 6)
    compilation.loc[compilation['CAR_DIST_KM'] < 2, 'CAR_CAB_FARE'] = 6
    compilation.loc[compilation['CAR_DIST_KM'] > 45, 'CAR_CAB_FARE'] = 50
    compilation['CAR_CAB_FARE'] = np.round(compilation['CAR_CAB_FARE']).clip(lower=8)
    compilation['CAR_FUEL_COST'] = compilation['CAR_DIST_KM'] * 0.25

    compilation['PT_FARE_COST'] = np.round(((compilation['PT_IVT'] - compilation['PT_IVT'].min()) / (
            compilation['PT_IVT'].max() - compilation['PT_IVT'].min())) * (3.5 - 1.09) + 1.09)

    # PT minimum fare = $1.09-$2.97 (bus); $1.09 - $2.50 (train)
    compilation['PT_FARE_COST'] = np.where(compilation['PT_NUM_MODES'] == {"SUBWAY", "BUS"},
                                                ((compilation['PT_IVT'] - compilation['PT_IVT'].min()) / (
                                                        compilation['PT_IVT'].max() - compilation[
                                                    'PT_IVT'].min())) * (5.47 - 2.18) + 2.18,
                                                np.where(compilation['PT_NUM_MODES'] == {"BUS"},
                                                         ((compilation['PT_IVT'] - compilation[
                                                             'PT_IVT'].min()) / (
                                                                  compilation['PT_IVT'].max() - compilation[
                                                              'PT_IVT'].min())) * (2.97 - 1.09) + 1.09,
                                                         np.where(compilation['PT_NUM_MODES'] == {"SUBWAY"},
                                                                  ((compilation['PT_IVT'] - compilation[
                                                                      'PT_IVT'].min()) / (
                                                                           compilation['PT_IVT'].max() -
                                                                           compilation[
                                                                               'PT_IVT'].min())) * (2.50 - 1.09) + 1.09,
                                                                  ((compilation['PT_IVT'] - compilation[
                                                                      'PT_IVT'].min()) / (
                                                                           compilation['PT_IVT'].max() -
                                                                           compilation[
                                                                               'PT_IVT'].min())) * (6 - 1.09) + 1.09)))


    compilation['PT_OVT_PRP'] = compilation['PT_OVT'] / compilation['PT_IVT']
    compilation['BIKE_PARKING_TIME'] = 5
    compilation['BIKE_OVT_PRP'] = compilation['BIKE_PARKING_TIME']/compilation['BIKE_TT']
    compilation['CAR_PARKING_TIME'] = 10  # Eventually can change to be a function of car park availability
    compilation['CAR_OVT_PRP'] = compilation['CAR_PARKING_TIME'] / compilation['CAR_TT']

    compilation['PT_TT'] = compilation['PT_IVT'] + compilation['PT_OVT']

    compilation['BIKE_FASTER'] = (compilation['PT_TT'] - compilation['BIKE_TT']) / (
                compilation['PT_TT'] + compilation['BIKE_TT'])
    compilation['WALK_FASTER'] = (compilation['PT_TT'] - compilation['WALK_TT']) / (
                compilation['PT_TT'] + compilation['WALK_TT'])
    compilation['CAR_FASTER'] = (compilation['PT_TT'] - compilation['CAR_TT']) / (
                compilation['PT_TT'] + compilation['CAR_TT'])

    compilation['SHORT_BIKE_ADVANTAGE'] = [1 if x <= 2 else 0 for x in compilation['BIKE_DIST_KM']]
    compilation['SHORT_WALK_ADVANTAGE'] = [1 if x <= 15 else 0 for x in compilation['WALK_TT']]

    compilation = compilation.merge(parking_costs, left_on="DEST_MTZ_1", right_on="SUB_MTZ_NO", how='left'
                                    ).drop(columns=['SUB_MTZ_NO', 'PT_TT', 'PT_NUM_MODES'])

    return compilation

def od_trips_by_mode_aggregated(trip_makers, mode_choice_attributes, choice_coefficients, zone_codes):
    # Updated 18/2/2025:
    T_ij_long = trip_makers.merge(mode_choice_attributes, on=['ORIGIN_MTZ_1', 'DEST_MTZ_1'], how='left')

    T_ij_long['CAR_PARK_COST'] = np.where(T_ij_long['TRIP_PARKING'] == '2HR', T_ij_long['avg_2HOURCOST'],
                                          np.where(T_ij_long['TRIP_PARKING'] == 'DAILY', T_ij_long['avg_SEASONCOST'], 0))

    T_ij_long[['P_car', 'P_cycle', 'P_pt', 'P_walk', 'P_taxi']] = T_ij_long.apply(
        lambda x: pd.Series(mode_choices_probabilities(x, choice_coefficients)), axis=1)

    T_ij_grouped = T_ij_long.groupby(['ORIGIN_MTZ_1', 'DEST_MTZ_1']).agg(
        {"PAX_ID": "count", "P_car": "mean", "P_cycle": "mean", "P_pt": "mean", "P_walk": "mean", "P_taxi": "mean"})
    T_ij_grouped = T_ij_grouped.rename(columns={"PAX_ID": "Trips"})

    for mode in ['car', 'cycle', 'pt', 'walk', 'taxi']:
        T_ij_grouped[f'Trips_{mode}'] = T_ij_grouped['Trips'] * T_ij_grouped[f'P_{mode}']

    T_ij_grouped[['Trips', 'Trips_car', 'Trips_cycle', 'Trips_pt', 'Trips_walk', 'Trips_taxi']] = adjust_trip_split(
        T_ij_grouped[['Trips', 'Trips_car', 'Trips_cycle', 'Trips_pt', 'Trips_walk', 'Trips_taxi']])

    trips = {}
    for mode in ['car', 'cycle', 'pt', 'walk', 'taxi']:
        mat = T_ij_grouped[[f'Trips_{mode}']].reset_index().pivot(index='ORIGIN_MTZ_1', columns='DEST_MTZ_1'
                                                                  ).fillna(0)
        mat.columns = mat.columns.droplevel(0)
        trips[mode.capitalize()] = order_impedance_matrices(mat, zone_codes)

    return trips, T_ij_grouped
def mode_choices_probabilities(row, choice_coefficients):
    # Update 18/2/2025: Calculate mode choice probabilities and derive trips from average probabilities.
    if row['HH_CAR_OWNERSHIP'] == 1:
        v_car = choice_coefficients['asc_car'] + \
                choice_coefficients['b_ivt'] * row['CAR_TT'] + \
                choice_coefficients['b_ovt'] * row['CAR_OVT_PRP'] + \
                choice_coefficients['b_multCars'] * row['HH_MULT_CARS']+ \
                choice_coefficients['b_fasterTrip'] * row['CAR_FASTER'] + \
                choice_coefficients['b_oopcosts'] * (row['CAR_PARK_COST'] + row['CAR_FUEL_COST']) +\
                choice_coefficients['b_cost_highincome'] * row['HIGH_INCOME'] * (row['CAR_PARK_COST'] + row['CAR_FUEL_COST'])
    else:
        v_car = 0

    v_cycle = (choice_coefficients['asc_cycle']) + \
              choice_coefficients['b_ivt'] * row['BIKE_TT'] + \
              choice_coefficients['b_ovt'] * row['BIKE_OVT_PRP'] + \
              choice_coefficients['b_multBikes'] * row['HH_MULT_BIKES'] + \
              choice_coefficients['b_shortTrip'] * row['SHORT_BIKE_ADVANTAGE'] + \
              choice_coefficients['b_fasterTrip'] * row['BIKE_FASTER']
    v_pt = choice_coefficients['b_ivt'] * row['PT_IVT'] + \
           choice_coefficients['b_ovt'] * row['PT_OVT_PRP'] + \
           choice_coefficients['b_oopcosts'] * row['PT_FARE_COST'] + \
           choice_coefficients['b_cost_highincome'] * row['HIGH_INCOME'] * row['PT_FARE_COST']
    v_walk = choice_coefficients['asc_walk'] + \
             choice_coefficients['b_ivt'] * row['WALK_TT'] + \
             choice_coefficients['b_shortTrip'] * row['SHORT_WALK_ADVANTAGE'] + \
             choice_coefficients['b_fasterTrip'] * row['WALK_FASTER']
    v_taxi = choice_coefficients['asc_taxi'] + \
             choice_coefficients['b_ivt'] * row['CAR_TT'] + \
             choice_coefficients['b_fasterTrip'] * row['CAR_FASTER'] + \
             choice_coefficients['b_oopcosts'] * row['CAR_CAB_FARE'] + \
             choice_coefficients['b_cost_highincome'] * row['HIGH_INCOME'] * row['CAR_CAB_FARE']


    P_car = np.exp(v_car) / (np.exp(v_car) + np.exp(v_cycle) + np.exp(v_pt) + np.exp(v_taxi) + np.exp(v_walk))
    P_cycle = np.exp(v_cycle) / (np.exp(v_car) + np.exp(v_cycle) + np.exp(v_pt) + np.exp(v_taxi) + np.exp(v_walk))
    P_pt = np.exp(v_pt) / (np.exp(v_car) + np.exp(v_cycle) + np.exp(v_pt) + np.exp(v_taxi) + np.exp(v_walk))
    P_taxi = np.exp(v_taxi) / (np.exp(v_car) + np.exp(v_cycle) + np.exp(v_pt) + np.exp(v_taxi) + np.exp(v_walk))
    P_walk = np.exp(v_walk) / (np.exp(v_car) + np.exp(v_cycle) + np.exp(v_pt) + np.exp(v_taxi) + np.exp(v_walk))

    probabilities = [P_car, P_cycle, P_pt, P_walk, P_taxi]

    return probabilities
def adjust_trip_split(df):
    trip_columns = ['Trips_car', 'Trips_cycle', 'Trips_pt', 'Trips_walk', 'Trips_taxi']

    # Round the trip columns
    df[trip_columns] = np.round(df[trip_columns])

    # Calculate the initial sum of the trip columns
    trip_sum = df[trip_columns].sum(axis=1)
    difference = df['Trips'] - trip_sum

    # Handle rows where difference is +1
    add_mask = difference == 1
    if add_mask.any():
        # Find the column with the lowest non-zero value for these rows
        min_columns = df.loc[add_mask, trip_columns].replace(0, np.inf).idxmin(axis=1)
        # Add 1 to those columns
        adjustment_matrix = pd.get_dummies(min_columns).reindex(columns=trip_columns, fill_value=0)
        df.loc[add_mask, trip_columns] += adjustment_matrix

    # Handle rows where difference is -1
    subtract_mask = difference == -1
    if subtract_mask.any():
        # Find the column with the highest non-zero value for these rows
        max_columns = df.loc[subtract_mask, trip_columns].replace(0, -np.inf).idxmax(axis=1)
        # Subtract 1 from those columns
        adjustment_matrix = pd.get_dummies(max_columns).reindex(columns=trip_columns, fill_value=0)
        df.loc[subtract_mask, trip_columns] -= adjustment_matrix

    return df

def od_trips_by_mode_disaggregated(T_ij, mode_choice_attributes, mc_coefficients, zone_codes):
    # The disaggregated approach takes the synthetic population, for which origins/destinations/ownership status are known.
    # 1. The mode choice attributes are combined with the synthetic population dataframe (each row a trip).
    # 2. The mode choice probabilities are calculated for each trip.
    # 3. The highest probability mode is taken as the mode choice for each trip
    T_ij_long = T_ij.copy()
    T_ij_long = T_ij_long.merge(mode_choice_attributes, on=['ORIGIN_MTZ_1', 'DEST_MTZ_1'], how='left')

    T_ij_long['ownership_type'] = T_ij_long.apply(lambda row: 'CarOnly' if ((row['HH_CAR_OWNERSHIP'] == 1) & (row['HH_BIKE_OWNERSHIP'] == 0)) else 'BikeOnly' if ((row['HH_CAR_OWNERSHIP'] == 0) & (row['HH_BIKE_OWNERSHIP'] == 1))
                                                  else 'OwnsNeither' if ((row['HH_CAR_OWNERSHIP'] == 0) & (row['HH_BIKE_OWNERSHIP'] == 0)) else 'OwnsBoth', axis=1)

    T_ij_long['ModeChoice'] = T_ij_long.apply(
        lambda row: individual_mode_choices(row, mc_coefficients), axis=1)

    trips = {}
    trips['PT'] = order_impedance_matrices(
        T_ij_long[T_ij_long['ModeChoice'] == 'PT'][['ORIGIN_MTZ_1', 'DEST_MTZ_1', 'ModeChoice']].groupby(
            ['ORIGIN_MTZ_1', 'DEST_MTZ_1']).count().reset_index().pivot(index='ORIGIN_MTZ_1', columns='DEST_MTZ_1',
                                                                        values='ModeChoice').fillna(0), zone_codes)

    trips['Cycle'] = order_impedance_matrices(
        T_ij_long[T_ij_long['ModeChoice'] == 'Cycle'][['ORIGIN_MTZ_1', 'DEST_MTZ_1', 'ModeChoice']].groupby(
            ['ORIGIN_MTZ_1', 'DEST_MTZ_1']).count().reset_index().pivot(index='ORIGIN_MTZ_1', columns='DEST_MTZ_1',
                                                                        values='ModeChoice').fillna(0), zone_codes)

    trips['Car'] = order_impedance_matrices(
        T_ij_long[T_ij_long['ModeChoice'] == 'Car'][['ORIGIN_MTZ_1', 'DEST_MTZ_1', 'ModeChoice']].groupby(
            ['ORIGIN_MTZ_1', 'DEST_MTZ_1']).count().reset_index().pivot(index='ORIGIN_MTZ_1', columns='DEST_MTZ_1',
                                                                        values='ModeChoice').fillna(0), zone_codes)
    return trips, T_ij_long
def individual_mode_choices(row, choice_coefficients):
    v_car = choice_coefficients['asc_car'] + \
                choice_coefficients['b_ivt'] * row['CAR_TT'] + \
                choice_coefficients['b_ovt'] * row['CAR_OVT_PRP'] + \
                choice_coefficients['b_multCars'] * row['HH_MULT_CARS']+ \
                choice_coefficients['b_fasterTrip'] * row['CAR_FASTER'] + \
                choice_coefficients['b_oopcosts'] * (row['CAR_PARK_COST'] + row['CAR_FUEL_COST']) +\
                choice_coefficients['b_cost_highincome'] * row['HIGH_INCOME'] * (row['CAR_PARK_COST'] + row['CAR_FUEL_COST'])
    v_cycle = (choice_coefficients['asc_cycle']) + \
              choice_coefficients['b_ivt'] * row['BIKE_TT'] + \
              choice_coefficients['b_ovt'] * row['BIKE_OVT_PRP'] + \
              choice_coefficients['b_multBikes'] * row['HH_MULT_BIKES'] + \
              choice_coefficients['b_shortTrip'] * row['SHORT_BIKE_ADVANTAGE'] + \
              choice_coefficients['b_fasterTrip'] * row['BIKE_FASTER']
    v_pt = choice_coefficients['b_ivt'] * row['PT_IVT'] + \
           choice_coefficients['b_ovt'] * row['PT_OVT_PRP'] + \
           choice_coefficients['b_oopcosts'] * row['PT_FARE_COST'] + \
           choice_coefficients['b_cost_highincome'] * row['HIGH_INCOME'] * row['PT_FARE_COST']
    v_walk = choice_coefficients['asc_walk'] + \
             choice_coefficients['b_ivt'] * row['WALK_TT'] + \
             choice_coefficients['b_shortTrip'] * row['SHORT_WALK_ADVANTAGE'] + \
             choice_coefficients['b_fasterTrip'] * row['WALK_FASTER']
    v_taxi = choice_coefficients['asc_taxi'] + \
             choice_coefficients['b_ivt'] * row['CAR_TT'] + \
             choice_coefficients['b_fasterTrip'] * row['CAR_FASTER'] + \
             choice_coefficients['b_oopcosts'] * row['CAR_CAB_FARE'] + \
             choice_coefficients['b_cost_highincome'] * row['HIGH_INCOME'] * row['CAR_CAB_FARE']

    if row['HH_CAR_OWNERSHIP'] == 1:
        P_car = np.exp(v_car) / (np.exp(v_car) + np.exp(v_cycle) + np.exp(v_pt) + np.exp(v_taxi) + np.exp(v_walk))
    else:
        P_car = 0

    P_cycle = np.exp(v_cycle) / (np.exp(v_car) + np.exp(v_cycle) + np.exp(v_pt) + np.exp(v_taxi) + np.exp(v_walk))
    P_pt = np.exp(v_pt) / (np.exp(v_car) + np.exp(v_cycle) + np.exp(v_pt) + np.exp(v_taxi) + np.exp(v_walk))
    P_taxi = np.exp(v_taxi) / (np.exp(v_car) + np.exp(v_cycle) + np.exp(v_pt) + np.exp(v_taxi) + np.exp(v_walk))
    P_walk = np.exp(v_walk) / (np.exp(v_car) + np.exp(v_cycle) + np.exp(v_pt) + np.exp(v_taxi) + np.exp(v_walk))

    probabilities = {'Car': P_car, 'Cycle': P_cycle, 'PT': P_pt, 'Walk':P_walk, 'Taxi':P_taxi}
    choice = max(probabilities, key=probabilities.get)

    return choice

def expand_dataframe(input_df):
    # Define mapping for ownership status
    df = input_df.copy()
    ownership_status_mapping = {
        "Trips_BikeOnly": {"HH_CAR_OWNERSHIP": 0, "HH_BIKE_OWNERSHIP": 1},
        "Trips_CarOnly": {"HH_CAR_OWNERSHIP": 1, "HH_BIKE_OWNERSHIP": 0},
        "Trips_OwnsBoth": {"HH_CAR_OWNERSHIP": 1, "HH_BIKE_OWNERSHIP": 1},
        "Trips_OwnsNeither": {"HH_CAR_OWNERSHIP": 0, "HH_BIKE_OWNERSHIP": 0}}

    trips_columns = ["Trips_BikeOnly", "Trips_CarOnly", "Trips_OwnsBoth", "Trips_OwnsNeither"]
    df_melted = df.melt(
        id_vars=["ORIGIN_MTZ_1", "DEST_MTZ_1", "CarOnlyMC", "BikeOnlyMC", "OwnsBothMC", "OwnsNeitherMC"],
        value_vars=trips_columns,
        var_name="OwnershipCategory",
        value_name="TripCount")

    # Drop rows where TripCount is zero or NaN
    df_melted = df_melted[df_melted["TripCount"] > 0].reset_index(drop=True)

    # Repeat rows based on TripCount
    df_melted = df_melted.loc[df_melted.index.repeat(df_melted["TripCount"].astype(int))].reset_index(drop=True)

    # Add ownership columns based on the ownership mapping
    df_melted["HH_CAR_OWNERSHIP"] = df_melted["OwnershipCategory"].map(lambda x: ownership_status_mapping[x]["HH_CAR_OWNERSHIP"])
    df_melted["HH_BIKE_OWNERSHIP"] = df_melted["OwnershipCategory"].map(lambda x: ownership_status_mapping[x]["HH_BIKE_OWNERSHIP"])

    # Dynamically assign the ModeChoice column based on the corresponding mode choice column
    def get_mode_choice(row):
        if row["OwnershipCategory"] == "Trips_BikeOnly":
            return row["BikeOnlyMC"]
        elif row["OwnershipCategory"] == "Trips_CarOnly":
            return row["CarOnlyMC"]
        elif row["OwnershipCategory"] == "Trips_OwnsBoth":
            return row["OwnsBothMC"]
        elif row["OwnershipCategory"] == "Trips_OwnsNeither":
            return row["OwnsNeitherMC"]
        return None

    df_melted["ModeChoice"] = df_melted.apply(get_mode_choice, axis=1)

    # Select and reorder the columns for the final output
    final_columns = ["ORIGIN_MTZ_1", "DEST_MTZ_1", "HH_CAR_OWNERSHIP", "HH_BIKE_OWNERSHIP", "ModeChoice"]
    return df_melted[final_columns]

def mode_split_aggregated(travel_times, T_ij,
                CO_time_coefficient = -0.0004, NCO_time_coefficient = -0.0003,
                CO_constant_CAR = 0.23, CO_constant_BIKE = - 3.1, NCO_constant_BIKE = -2.51):
    
    #######################################################################################
    # INPUTS                                                                              #
    # travel_times = dict with keys as mode names and values as mode travel time matrix.  #
    # T_ij = dict with keys as vehicle ownership status and values as OD matrix           #
    #                                                                                     #
    # Coefficients and constants are derived from a mode choice model.                    #
    # CO = Car-Owners; NCO = Non-Car-Owners                                               #
    # #####################################################################################

    V_ij_CAR_CO = travel_times['DRIVE'] * CO_time_coefficient + CO_constant_CAR # Car
    V_ij_BIKE_CO = travel_times['BIKE'] * CO_time_coefficient + CO_constant_BIKE # Bike
    V_ij_WALK_CO = travel_times['WALK'] * CO_time_coefficient

    CO_V_sum = np.exp(V_ij_CAR_CO) + np.exp(V_ij_BIKE_CO) + np.exp(V_ij_WALK_CO)

    P_CAR_CO = np.exp(V_ij_CAR_CO) / CO_V_sum
    P_BIKE_CO = np.exp(V_ij_BIKE_CO) / CO_V_sum
    P_WALK_CO = np.exp(V_ij_WALK_CO) / CO_V_sum

    T_ij_DRIVE_CO = T_ij['CAR_OWNER'] * P_CAR_CO
    T_ij_BIKE_CO = T_ij['CAR_OWNER'] * P_BIKE_CO
    T_ij_WALK_CO = T_ij['CAR_OWNER'] * P_WALK_CO

    V_ij_BIKE_NCO = travel_times['BIKE'] * NCO_time_coefficient + NCO_constant_BIKE # Bike
    V_ij_WALK_NCO = travel_times['WALK'] * NCO_time_coefficient
    NCO_V_sum = np.exp(V_ij_BIKE_NCO) + np.exp(V_ij_WALK_NCO)
    P_BIKE_NCO = np.exp(V_ij_BIKE_NCO) / NCO_V_sum
    P_WALK_NCO = np.exp(V_ij_WALK_NCO) / NCO_V_sum

    T_ij_BIKE_NCO = T_ij['NON_CAR_OWNER'] * P_BIKE_NCO
    T_ij_WALK_NCO = T_ij['NON_CAR_OWNER'] * P_WALK_NCO

    od_matrices = {}
    od_matrices['DRIVE'] = T_ij_DRIVE_CO
    od_matrices['BIKE'] = T_ij_BIKE_CO + T_ij_BIKE_NCO
    od_matrices['WALK'] = T_ij_WALK_CO + T_ij_WALK_NCO

    probabilities = {}
    _ = {}
    _['DRIVE'] = P_CAR_CO
    _['BIKE'] = P_BIKE_CO
    _['WALK'] = P_WALK_CO
    probabilities['CAR_OWNER'] = _

    _ = {}
    _['BIKE'] = P_BIKE_NCO
    _['WALK'] = P_WALK_NCO
    probabilities['NON_CAR_OWNER'] = _

    for mode in od_matrices.keys():
        matrix = od_matrices[mode]
        matrix = np.round(matrix).astype(int)
        od_matrices[mode] = matrix

    return od_matrices