import pickle
import pandas as pd
import numpy as np


def order_impedance_matrices(impedance_matrix, zone_codes):
    # This is to ensure that the null zones are removed and that the matrices are ordered correctly.
    # Order is important because sometimes dataframes or arrays are merged with pd.concat or pd.join instead of pd.merge on key
    impedance_matrix = impedance_matrix.astype(float)
    impedance_matrix = impedance_matrix.reindex(index=zone_codes, columns=zone_codes, fill_value=0)
    # impedance_matrix = impedance_matrix[zone_codes]
    # impedance_matrix = impedance_matrix.loc[zone_codes]
    # impedance_matrix = impedance_matrix[sorted(zone_codes)].sort_index()
    return impedance_matrix

def adjust_row_for_difference(T_ij_matrix, original_values, row_index, target_sum):
    """
    Adjusts row `row_index` in `T_ij_matrix` to meet `target_sum` by efficiently spreading the `difference`
    across multiple cells that originally had non-zero values.
    Prioritizes adjustments by adding to the smallest values first and reducing the largest values first.
    """
    row_sum = T_ij_matrix[row_index, :].sum()
    difference = target_sum - row_sum

    # Get indices of cells that originally had a non-zero value
    eligible_indices = np.where(original_values[row_index, :] > 0)[0]

    # Sort eligible cells once
    if difference > 0:
        # Sort in ascending order to add to the smallest values first
        sorted_indices = eligible_indices[np.argsort(T_ij_matrix[row_index, eligible_indices])]
    else:
        # Sort in descending order to reduce the largest values first
        sorted_indices = eligible_indices[np.argsort(-T_ij_matrix[row_index, eligible_indices])]

    if len(sorted_indices) != 0 and difference != 0:
        # Distribute the difference in chunks rather than iterating one by one
        while abs(np.round(difference)) > 0:
            # Determine the maximum adjustment we can apply to each cell in this pass
            adjustment = np.sign(difference)  # Either +1 or -1

            # Calculate how much to adjust each selected cell
            if abs(difference) < len(sorted_indices):
                sorted_indices = sorted_indices[:int(np.ceil(abs(difference)))]
                per_cell_adjustment = 1
            else:
                per_cell_adjustment = abs(difference) // len(sorted_indices)

            # Calculate the total adjustment for this pass
            total_adjustment = int(adjustment * per_cell_adjustment)

            # Apply adjustments in batch to each selected cell
            T_ij_matrix[row_index, sorted_indices] += total_adjustment
            difference -= (total_adjustment * len(sorted_indices))
            # print("in while loop", difference, abs(np.round(difference)))
    return T_ij_matrix

def trip_distribution(car_ownership, P, A, travel_times, network_mode_split, n, b, zone_codes):
    zones = len(P)
    car_ownership = car_ownership.tolist()

    # Car-owning households trip distribution
    P_CO = P * car_ownership
    composite_impedance_CO = (network_mode_split.loc['DRIVE', 'CAR_OWNER'] * travel_times['DRIVE'] +
                              network_mode_split.loc['BIKE', 'CAR_OWNER'] * travel_times['BIKE'] +
                              network_mode_split.loc['WALK', 'CAR_OWNER'] * travel_times['WALK'])

    T_ij_CO = np.zeros((zones, zones))
    for i in range(zones):
        for j in range(zones):
            if i != j:
                T_ij_CO[i, j] = P_CO.iloc[i] * A.iloc[j] * composite_impedance_CO.iloc[i, j] ** n * np.exp(
                    -b * (composite_impedance_CO.iloc[i, j]))

    # Iterative balancing for Car-Owners
    for iteration in range(100):
        for i in range(zones):
            row_sum = np.nansum(T_ij_CO[i, :])
            if row_sum > 0:
                T_ij_CO[i, :] *= P_CO.iloc[i] / row_sum
    # Store original values for later adjustment
    original_T_ij_CO = T_ij_CO.copy()

    # Round trips to integers and apply adjustments to match row totals
    T_ij_CO = np.round(T_ij_CO).astype(int)
    for i in range(zones):
        T_ij_CO = adjust_row_for_difference(T_ij_CO, original_T_ij_CO, i, P_CO.iloc[i])

    # Non-car-owning households trip distribution
    P_NCO = P * [1-x for x in car_ownership]
    composite_impedance_NCO = (network_mode_split.loc['DRIVE', 'NON_CAR_OWNER'] * travel_times['DRIVE'] +
                              network_mode_split.loc['BIKE', 'NON_CAR_OWNER'] * travel_times['BIKE'] +
                              network_mode_split.loc['WALK', 'NON_CAR_OWNER'] * travel_times['WALK'])

    T_ij_NCO = np.zeros((zones, zones))
    for i in range(zones):
        for j in range(zones):
            if i != j:
                T_ij_NCO[i, j] = P_NCO.iloc[i] * A.iloc[j] * composite_impedance_NCO.iloc[i, j] ** n * np.exp(
                    -b * (composite_impedance_NCO.iloc[i, j]))

    # Iterative balancing for Non-Car-Owners
    for iteration in range(100):
        for i in range(zones):
            row_sum = np.nansum(T_ij_NCO[i, :])
            if row_sum > 0:
                T_ij_NCO[i, :] *= P_NCO.iloc[i] / row_sum

    # Store original values for later adjustment
    original_T_ij_NCO = T_ij_NCO.copy()

    # Round trips to integers and apply adjustments to match row totals
    T_ij_NCO = np.round(T_ij_NCO).astype(int)
    for i in range(zones):
        T_ij_NCO = adjust_row_for_difference(T_ij_NCO, original_T_ij_NCO, i, P_NCO.iloc[i])

    # Combine the matrices if needed for overall validation
    T_total = T_ij_CO + T_ij_NCO

    # Scaling to ensure combined totals match original P, if needed
    scaling_factor = P.sum() / T_total.sum()
    T_ij_CO = (T_ij_CO * scaling_factor).round().astype(int)
    T_ij_NCO = (T_ij_NCO * scaling_factor).round().astype(int)

    T_ij_NCO = pd.DataFrame(T_ij_NCO, index=zone_codes, columns=zone_codes)
    T_ij_CO = pd.DataFrame(T_ij_CO, index=zone_codes, columns=zone_codes)

    T_ij = {'CAR_OWNER':T_ij_CO, 'NON_CAR_OWNER':T_ij_NCO}

    return T_ij

def trip_distribution_superseded(car_ownership, P, A, mode_A_time, mode_B_time, mode_C_time, network_mode_split, n, b,
                      mode_A='Car', mode_B='Cycle', mode_C='PT', X_label='CarOwners', Y_label='NonCarOwners'):

    ###############################################################################
    #                                  DO NOT USE                                 #
    # An approach that does not require ownership- and mode-specific coefficients #
    #                 Correctness of this approach is questionable                #
    ###############################################################################

    zones = len(P)

    # Car-owning households trip distribution
    P_X = P * car_ownership
    composite_impedance_X = (network_mode_split.loc[mode_A, 'CAR_MODE_SPLIT'] * mode_A_time +
                             network_mode_split.loc[mode_B, 'CAR_MODE_SPLIT'] * mode_B_time +
                             network_mode_split.loc[mode_C, 'CAR_MODE_SPLIT'] * mode_C_time)

    T_ij_X = np.zeros((zones, zones))
    for i in range(zones):
        for j in range(zones):
            if i != j:
                T_ij_X[i, j] = P_X.iloc[i] * A.iloc[j] * composite_impedance_X.iloc[i, j] ** n * np.exp(
                    -b * (composite_impedance_X.iloc[i, j]))

    # Iterative balancing to ensure convergence
    for iteration in range(100):
        # Row adjustment (production side)
        for i in range(zones):
            row_sum = np.nansum(T_ij_X[i, :])
            if row_sum > 0:
                T_ij_X[i, :] *= P_X.iloc[i] / row_sum

    # Non-car-owning households trip distribution
    P_Y = P * (1 - car_ownership)
    composite_impedance_Y = (network_mode_split.loc[mode_A, 'NONCAR_MODE_SPLIT'] * mode_A_time +
                             network_mode_split.loc[mode_B, 'NONCAR_MODE_SPLIT'] * mode_B_time +
                             network_mode_split.loc[mode_C, 'NONCAR_MODE_SPLIT'] * mode_C_time)

    T_ij_Y = np.zeros((zones, zones))
    for i in range(zones):
        for j in range(zones):
            if i != j:
                T_ij_Y[i, j] = P_Y.iloc[i] * A.iloc[j] * composite_impedance_Y.iloc[i, j] ** n * np.exp(
                    -b * (composite_impedance_Y.iloc[i, j]))

    # Iterative balancing to ensure convergence
    for iteration in range(100):
        # Row adjustment (production side)
        for i in range(zones):
            row_sum = np.nansum(T_ij_Y[i, :])
            if row_sum > 0:
                T_ij_Y[i, :] *= P_Y.iloc[i] / row_sum

    aggregate_trips = {}
    aggregate_trips[X_label] = T_ij_X
    aggregate_trips[Y_label] = T_ij_Y

    return aggregate_trips
