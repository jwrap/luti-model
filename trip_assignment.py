import pandas as pd
import geopandas as gpd
import networkx as nx
import itertools, warnings, os, pickle, sys, gc
import pandana as pdna
warnings.filterwarnings("ignore")
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed

# Code from https://github.com/nlperic/ta-lab
from ta_lab.assignment.line import *
from ta_lab.assignment.graph import *

def generate_aon_paths(g_df, node_df, subzone_nodes_df, network_twoway=False):
    subzone_nodes_df.columns = ['ZONE_CODE', 'closest_nodeID']
    od_nodes = subzone_nodes_df['closest_nodeID'].tolist()
    perms = [perm for perm in itertools.permutations(od_nodes, 2)]
    origins = [o for o, d in perms]
    destinations = [d for o, d in perms]

    node_df = node_df.set_index('nodeID')

    # Initialize network
    net = pdna.Network(node_df["x"], node_df["y"], g_df["source"], g_df["target"],
                                             g_df[["time"]], twoway=network_twoway)

    paths = net.shortest_paths(origins, destinations, imp_name='time')
    paths_dict = dict(zip(tuple(zip(origins, destinations)), paths))

    return paths_dict

def generate_aon_impedance_matrix(g_df, node_df, subzone_nodes_df, imp_name='time', network_twoway=False):
    if subzone_nodes_df is pd.DataFrame():
        subzone_nodes_df.columns = ['ZONE_CODE', 'nodeID']
        od_nodes = subzone_nodes_df['nodeID'].tolist()
        zone_node_map = dict(zip(subzone_nodes_df['nodeID'], subzone_nodes_df['ZONE_CODE']))
    else:
        od_codes = subzone_nodes_df.copy()
        od_nodes = [f'MTZ{int(x)}' for x in od_codes]
        zone_node_map = dict(zip(od_nodes, od_codes))

    if imp_name == 'length':
        imp_title = ""
    else:
        imp_title = "Time"

    perms = [perm for perm in itertools.permutations(od_nodes, 2)]
    origins = [o for o, d in perms]
    destinations = [d for o, d in perms]
    node_df = node_df.drop_duplicates('nodeID')
    node_df = node_df.set_index('nodeID')
    g_df[imp_name] = g_df[imp_name].clip(lower=1e-4, upper=500)
    g_df = g_df[g_df['source'].isin(node_df.index) & g_df['target'].isin(
        node_df.index)]

    # Pandana shortest_path calculations go haywire when imp. values are too small.
    # Use "amplifier" as an ad-hoc solution to avoid computation errors
    # Initialize network
    net = pdna.Network(node_df["x"], node_df["y"], g_df["source"], g_df["target"],
                       g_df[[imp_name]], twoway=network_twoway)

    # Get shortest (TIME) path lengths between all ODs for cars
    paths_lengths = net.shortest_path_lengths(origins, destinations, imp_name=imp_name)

    # Interzonal PT times only
    pt_od_long = pd.DataFrame()
    pt_od_long['Origin'] = [zone_node_map[x] for x in origins]
    pt_od_long['Destination'] = [zone_node_map[x] for x in destinations]
    pt_od_long[imp_title] = paths_lengths

    pt_od_paths = pt_od_long[['Origin', 'Destination', imp_title]]
    pt_od_wide = pt_od_paths.pivot(columns='Destination', index='Origin', values=imp_title)
    impedance_matrix = pt_od_wide[sorted(pt_od_wide.columns)].sort_index().fillna(0)

    impedance_matrix = impedance_matrix.astype(float)

    return impedance_matrix

def prepare_uncapped_network(network_edges_gdf, network_node_gdf, policy = False, base_capacity=10000000, alpha=0.15, beta=4):
    # Takes in nodes and edges dataframe of the network and returns columns necessary for trip assignment
    # Assume network lanes are all two-way
    # Assume no capacity constraints (assume capacity will not be a limiting factor)
    # Difference with prepare_capped_network is this function does not define link capacities

    if policy==True:
        mtz_nodes = network_edges_gdf[network_edges_gdf['ori_source'].str.contains("MTZ")]['ori_source']
        subset = network_edges_gdf[network_edges_gdf['ori_source'].isin(mtz_nodes)]
        nodeID_mapping = dict(zip(subset['ori_source'], subset['source']))
        g_df = network_edges_gdf[['source', 'target', 'length', 'time', 'edge', 'geometry']].copy()
        g_df['capacity'] = base_capacity  # aggregated demand into 2 hours
    else:
        network_node_gdf['IntIDs'] = network_node_gdf.index
        nodeID_mapping = dict(zip(network_node_gdf['nodeID'], network_node_gdf['IntIDs']))
        g_df = network_edges_gdf[['source', 'target', 'length', 'time', 'edge', 'geometry']].copy()
        g_df['source'] = g_df['source'].map(nodeID_mapping)
        g_df['target'] = g_df['target'].map(nodeID_mapping)
        g_df['capacity'] = base_capacity  # aggregated demand into 2 hours

    # FILL OTHER COLUMNS FOR TRIP ASSIGNMENT (per ta-lab code)
    g_df['alpha'] = alpha
    g_df['beta'] = beta

    return g_df, network_node_gdf, nodeID_mapping

def prepare_capped_network(network_edges_gdf, network_node_gdf, base_capacity, num_peak_hours=2, peak_hour_factor=0.95, alpha=0.15, beta=4):
    # Takes in nodes and edges dataframe of the network and returns columns necessary for trip assignment
    # Calculates link capacities based on speed limit, number of lanes, a speed adjustment factor, and a pre-defined base capacity.
    # Capacity calcs approach based on guideline in the Highway Performance Monitoring System (HPMS) Field Manual. Appendix N: Procedures for Estimating Highway Capacity. Federal Highway Administration
    # https://transops.s3.amazonaws.com/uploaded_files/LinkCapacityCalculation.pdf

    network_node_gdf['IntIDs'] = network_node_gdf.index
    nodeID_mapping = dict(zip(network_node_gdf['nodeID'], network_node_gdf['IntIDs']))

    g_df = network_edges_gdf[['source', 'target', 'maxspeed', 'highway', 'lanes', 'length', 'time', 'edge', 'geometry']].copy()
    g_df['source'] = g_df['source'].map(nodeID_mapping)
    g_df['target'] = g_df['target'].map(nodeID_mapping)
    g_df['base_capacity'] = base_capacity
    g_df['lanes'] = [np.floor(x) if x >= 5 else x for x in g_df['lanes']]
    g_df['speed_adj_factor'] = [7.2 if row['lanes'] <= 2 else 4.8 if row['lanes'] == 3 else 2.4 if row['lanes'] == 4 else 0 for _, row in g_df.iterrows()]
    g_df['FFS'] = g_df['maxspeed'] - g_df['speed_adj_factor']
    g_df['capacity_by_type'] = g_df['base_capacity'] + (10 * g_df['FFS'])
    g_df['peak_capacity'] = g_df['capacity_by_type'] * peak_hour_factor * g_df['lanes']
    g_df['capacity'] = g_df['peak_capacity'] * num_peak_hours  # aggregated demand into 2 hours

    # FILL OTHER COLUMNS FOR TRIP ASSIGNMENT (per ta-lab code)
    g_df['alpha'] = alpha
    g_df['beta'] = beta
    g_df = g_df.drop_duplicates(['source', 'target'])

    return g_df, network_node_gdf, nodeID_mapping

def process_chunk_combined_with_lpaths(chunk, tuple_edge_name, OD_json):

    ##########################################################
    # Some parallelization to speed up trip assignment       #
    # Currently unused as explode_edges_in_batches is faster #
    ##########################################################

    local_lpaths = {}  # Store lpaths for the batch
    local_potential_volume = defaultdict(float)  # Store potential_volume for the batch

    for perm, path in chunk:
        demand = OD_json.loc[perm[0], perm[1]]  # Get demand from OD matrix
        edges = []  # List of edges for this path

        # Convert path to edges and update potential_volume
        for i in range(len(path) - 1):
            edge = str(tuple_edge_name.get((int(path[i]), int(path[i + 1])), None))
            if edge is not None:
                edges.append(edge)
                local_potential_volume[edge] += demand

        # Store edges in local_lpaths
        local_lpaths[perm] = edges

    return local_potential_volume, local_lpaths

def compute_potential_volume_and_lpaths(perms_paths, tuple_edge_name, OD_json, batch_size=30000, n_workers=10):

    ##########################################################
    # Some parallelization to speed up trip assignment       #
    # Currently unused as explode_edges_in_batches is faster #
    ##########################################################

    # Split perms_paths into batches
    batches = [perms_paths[i:i + batch_size] for i in range(0, len(perms_paths), batch_size)]

    # Parallelize processing
    with Parallel(n_jobs=n_workers, backend='loky') as parallel:
        results = parallel(
            delayed(process_chunk_combined_with_lpaths)(batch, tuple_edge_name, OD_json)
            for batch in batches
        )

    # Merge results into global potential_volume and lpaths
    lpaths = {}
    potential_volume = defaultdict(float)

    for local_potential_volume, local_lpaths in results:
        # Merge potential volumes
        for key, value in local_potential_volume.items():
            potential_volume[key] += value
        # Merge lpaths
        lpaths.update(local_lpaths)

    return potential_volume, lpaths

def trip_assignment_uncapped(g_df, node_df, subzone_node_map, impedance_matrix, zone_codes, OD_json):

    #######################################################################################################################################################
    # INPUTS:                                                                                                                                             #
    # g_df: graph dataframe of network as produced by prepare_uncapped_network function                                                                   #
    # node_df: node dataframe of network as produced by prepare_uncapped_network function                                                                 #
    # subzone_node_map: mapping from string nodeID to integer nodeID as produced by prepare_uncapped_network function.                                    #
    # impedance_matrix: pre-calculated shortest path travel times (uncapped assumption means we can use precalculated (free-flow) travel times            #
    # zone_codes: list of zone codes, in 'correct' order                                                                                                  #
    # OD_json: trip OD matrices as produced in the preceding mode_split step                                                                              #
    #                                                                                                                                                     #
    # OUTPUTS:                                                                                                                                            #
    # vol_df: Volume (variable: demand) of traffic on each link in the network                                                                            #
    # travel_times: Shortest path travel time for each OD pair                                                                                            #
    # distance_df: Shortest path distance for each OD pair                                                                                                #
    #######################################################################################################################################################

    # The keys in this dictionary are meant to match the edge names used in the g_df source/target columns
    subzone_node_map = {float(k[3:]) if "MTZ" in k else float(k[-2:]): v for k, v in subzone_node_map.items() if "MTZ" in k or "CENTROID" in k}

    nt = Network('net')
    for _, row in g_df[['edge', 'source', 'target', 'time', 'capacity', 'alpha', 'beta']].iterrows():
        nt.add_edge(Edge([str(x) for x in row.tolist()]))

    perms = [perm for perm in itertools.permutations(zone_codes, 2)]
    origin_codes = [subzone_node_map[o] for o, d in perms]
    dest_codes = [subzone_node_map[d] for o, d in perms]

    g_df['time'] = g_df['time'].clip(lower=1e-4, upper=500)

    pdna_net = pdna.Network(node_df["x"], node_df["y"], g_df["source"], g_df["target"], g_df[['time', 'length']], twoway=False)

    paths = pdna_net.shortest_paths(origin_codes, dest_codes, imp_name='time')
    distances = pdna_net.shortest_path_lengths(origin_codes, dest_codes, imp_name='length')
    distance_df = pd.DataFrame()
    distance_df['Origin'] = [o for o, d in perms]
    distance_df['Destination'] = [d for o, d in perms]
    distance_df['Distance_M'] = distances

    tuple_edge_name = dict(zip(zip(g_df['source'].astype(int), g_df['target'].astype(int)), g_df['edge']))
    edge_map_df = pd.DataFrame(list(tuple_edge_name.items()), columns=["EDGES_TUPLES", "EDGE_NAME"])
    potential_volume = {link: 0 for link in nt.edgenode.values()}

    # perm_paths = list(zip(perms, paths))
    # potential_volume_, lpaths = compute_potential_volume_and_lpaths(
    #     perm_paths, tuple_edge_name, OD_json, batch_size=30000, n_workers=10)

    shortest_paths_df = pd.DataFrame({
        'ORIGIN_MTZ_1': [o for o, d in perms],
        'DEST_MTZ_1': [d for o, d in perms],
        'ROUTES': paths})
    shortest_paths_df = shortest_paths_df.merge(
        OD_json.melt(ignore_index=False).reset_index().rename(columns={"value": "DEMAND"}),
        on=['ORIGIN_MTZ_1', 'DEST_MTZ_1'], how='left')

    batch_size = 5000
    num_batches = (len(shortest_paths_df) // batch_size) + 1
    batches = [shortest_paths_df.iloc[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

    with Parallel(n_jobs=4, backend='loky') as parallel:
        results = parallel(delayed(explode_edges_in_batches)(batch) for batch in batches)

    # Combine all batch results into a single DataFrame
    master_df = pd.concat(results, ignore_index=True)
    master_df = master_df.groupby("EDGES_TUPLES", as_index=False)["DEMAND"].sum()
    master_df = master_df.merge(edge_map_df, on='EDGES_TUPLES', how='left').drop(columns=["EDGES_TUPLES"])
    potential_volume_ = master_df.set_index("EDGE_NAME").to_dict()['DEMAND']

    potential_volume.update(potential_volume_)

    travel_times = impedance_matrix.stack().reset_index().set_axis(['Origin', 'Destination', 'TT'], axis=1)

    vol2 = dict(potential_volume)
    vol_df = pd.DataFrame(index=vol2.keys(), columns=["demand"])
    vol_df["demand"] = vol2.values()
    vol_df.reset_index(inplace=True)
    vol_df.rename(columns={"index": "edge"}, inplace=True)
    vol_df = gpd.GeoDataFrame(vol_df.merge(g_df[['edge', 'geometry']], on='edge', how='left'))

    return vol_df, travel_times, distance_df

def explode_edges_in_batches(batch_df):
    ################################################################################
    # Converts routes to edges, aggregates demand, and returns the batch results.  #
    ################################################################################

    # Convert node sequences into edge tuples
    batch_df["EDGES_TUPLES"] = batch_df["ROUTES"].apply(lambda x: list(zip(x[:-1], x[1:])))

    # Explode to get one edge per row
    exploded_batch = batch_df.explode("EDGES_TUPLES")

    # Aggregate demand per edge tuple
    batch_travel_volume = exploded_batch.groupby("EDGES_TUPLES", as_index=False)["DEMAND"].sum()

    return batch_travel_volume

def trip_assignment_capped(g_df, node_df, subzone_node_map, zone_codes, OD_json, convergence_criteria=1, twoway=False):
    # 2025-03-15 Sped up with vectorised operations
    # Used for networks for which capacity is a concern

    #######################################################################################################################
    # INPUTS:
    # g_df: graph dataframe of network as produced by prepare_capped_network function
    # node_df: node dataframe of network as produced by prepare_capped_network function
    # subzone_node_map: mapping from string nodeID to integer nodeID as produced by prepare_capped_network function.
    # zone_codes: list of zone codes, in 'correct' order
    # OD_json: trip OD matrices as produced in the preceding mode_split step
    # convergence_criteria: lower = longer run, more stability in output
    # twoway: True assumes all of network permits two-way traffic; assume false for car network
    #
    # OUTPUTS:
    # vol_df: Volume (variable: demand) of traffic on each link in the network
    # travel_times: Shortest path travel time for each OD pair
    # distance_df: Shortest path distance for each OD pair
    #######################################################################################################################

    subzone_node_map = {float(k[3:]) if "MTZ" in k else float(k[-2:]): v for k, v in subzone_node_map.items() if "MTZ" in k or "CENTROID" in k}

    print('initializing network')
    nt = Network('net')

    for index, row in g_df[['edge', 'source', 'target', 'time', 'capacity', 'alpha', 'beta']].iterrows():
        nt.add_edge(Edge([str(x) for x in row.tolist()]))

    # initialize cost
    nt.init_cost()
    print('initialized network')
    empty = {link: 0 for link in nt.edgenode.values()}

    # initial all-or-nothing assignment (source: ta-lab)
    perms = [perm for perm in itertools.permutations(zone_codes, 2)]
    origin_codes = [subzone_node_map[o] for o, d in perms]
    dest_codes = [subzone_node_map[d] for o, d in perms]

    g_df['time'] = g_df['time'].clip(lower=1e-4, upper=500)
    pdna_net = pdna.Network(node_df["x"], node_df["y"], g_df["source"], g_df["target"], g_df[['time']], twoway=twoway)
    print('initialized pandana network')

    paths = pdna_net.shortest_paths(origin_codes, dest_codes, imp_name='time')
    tuple_edge_name = dict(zip(zip(g_df['source'].astype(int), g_df['target'].astype(int)), g_df['edge']))
    edge_map_df = pd.DataFrame(list(tuple_edge_name.items()), columns=["EDGES_TUPLES", "EDGE_NAME"])

    print('calculated shortest path lengths, now updating potential_volume')
    potential_volume = empty.copy()

    shortest_paths_df = pd.DataFrame({
        'ORIGIN_MTZ_1': [o for o, d in perms],
        'DEST_MTZ_1': [d for o, d in perms],
        'ROUTES': paths})
    shortest_paths_df = shortest_paths_df.merge(
        OD_json.melt(ignore_index=False).reset_index().rename(columns={"value": "DEMAND"}),
        on=['ORIGIN_MTZ_1', 'DEST_MTZ_1'], how='left')

    batch_size = 5000
    num_batches = (len(shortest_paths_df) // batch_size) + 1
    batches = [shortest_paths_df.iloc[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

    with Parallel(n_jobs=4, backend='loky') as parallel:
        results = parallel(delayed(explode_edges_in_batches)(batch) for batch in batches)

    # Combine all batch results into a single DataFrame
    master_df = pd.concat(results, ignore_index=True)
    master_df = master_df.groupby("EDGES_TUPLES", as_index=False)["DEMAND"].sum()
    master_df = master_df.merge(edge_map_df, on='EDGES_TUPLES', how='left').drop(columns=["EDGES_TUPLES"])
    potential_volume_ = master_df.set_index("EDGE_NAME").to_dict()['DEMAND']

    shortest_paths_df = shortest_paths_df.drop(columns=['DEMAND'])

    # perm_paths = list(zip(perms, paths))
    # potential_volume_, lpaths = compute_potential_volume_and_lpaths(
    #     perm_paths, tuple_edge_name, OD_json, batch_size=30000, n_workers=10)

    potential_volume.update(potential_volume_)
    volume = potential_volume.copy()
    potential_volume = empty.copy()
    temp_vol = empty.copy()
    step = 1

    x = 0
    while cal_limit(volume, temp_vol) > 0.5 and step > convergence_criteria:
        x += 1
        print(f'start of iteration: {x}')
        temp_vol = volume.copy()  # temp_vol is the old-vol used for cal_limit

        nt.update_cost(volume) # update travel costs on nt network that tracks travel costs.

        new_weights = [edge.cost for edge in nt.edgeset.values()]
        new_weights_df = pd.DataFrame(new_weights, columns=['time'])
        new_weights_df = pd.concat([new_weights_df, g_df['length']], axis=1)
        new_weights_df['time'] = new_weights_df['time'].clip(lower=1e-4, upper=500)

        pdna_net = pdna.Network(node_df["x"], node_df["y"], g_df["source"], g_df["target"],
                                    new_weights_df[['time']], twoway=twoway)

        paths = pdna_net.shortest_paths(origin_codes, dest_codes, imp_name='time')
        # perm_paths = list(zip(perms, paths))
        # potential_volume_, lpaths = compute_potential_volume_and_lpaths(
        #     perm_paths, tuple_edge_name, OD_json, batch_size=30000, n_workers=10)

        shortest_paths_df['paths'] = paths
        shortest_paths_df = shortest_paths_df.merge(
            OD_json.melt(ignore_index=False).reset_index().rename(columns={"value": "DEMAND"}),
            on=['ORIGIN_MTZ_1', 'DEST_MTZ_1'], how='left')

        batch_size = 5000
        num_batches = (len(shortest_paths_df) // batch_size) + 1
        batches = [shortest_paths_df.iloc[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

        with Parallel(n_jobs=4, backend='loky') as parallel:
            results = parallel(delayed(explode_edges_in_batches)(batch) for batch in batches)

        # Combine all batch results into a single DataFrame
        master_df = pd.concat(results, ignore_index=True)
        master_df = master_df.groupby("EDGES_TUPLES", as_index=False)["DEMAND"].sum()
        master_df = master_df.merge(edge_map_df, on='EDGES_TUPLES', how='left').drop(columns=["EDGES_TUPLES"])
        potential_volume_ = master_df.set_index("EDGE_NAME").to_dict()['DEMAND']

        potential_volume.update(potential_volume_)

        step = cal_step(nt, volume, potential_volume)

        for link in nt.edge_id_set:
            volume[link] = volume[link] + step * (potential_volume[link] - volume[link])

        potential_volume = empty.copy()

        print(f'end of iteration: {x} \t\t cal_limit: {cal_limit(volume, temp_vol)} \t\t step: {step}')

    pdna_net = pdna.Network(node_df["x"], node_df["y"], g_df["source"], g_df["target"],
                                    new_weights_df[['time', 'length']], twoway=twoway)

    costs = pdna_net.shortest_path_lengths(origin_codes, dest_codes, imp_name='time')
    travel_times = pd.DataFrame()
    travel_times['Origin'] = [o for o, d in perms]
    travel_times['Destination'] = [d for o, d in perms]
    travel_times['TT'] = costs

    distances = pdna_net.shortest_path_lengths(origin_codes, dest_codes, imp_name='length')
    distance_df = pd.DataFrame()
    distance_df['Origin'] = [o for o, d in perms]
    distance_df['Destination'] = [d for o, d in perms]
    distance_df['Distance_M'] = distances

    vol_df = pd.DataFrame(index=volume.keys(), columns=["demand"])
    vol_df["demand"] = volume.values()
    vol_df.reset_index(inplace=True)
    vol_df.rename(columns={"index": "edge"}, inplace=True)
    vol_df = gpd.GeoDataFrame(vol_df.merge(g_df[['edge', 'geometry', 'capacity']], on='edge', how='left'))
    vol_df['demand/capacity'] = vol_df['demand'] / vol_df['capacity']
    gc.collect()

    return vol_df, travel_times, distance_df