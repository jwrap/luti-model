import pickle
import pandas as pd
import itertools, momepy
import networkx as nx
import numpy as np
import scipy.spatial
from shapely.geometry import LineString
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from collections import defaultdict

def connect_subgraphs_bidirectional(edges_df, nodes_df, new_link_label, threshold=0.0004):
    # Step 1: Convert edges_df to a directed NetworkX graph
    G_bike = momepy.gdf_to_nx(edges_df, approach="primal", length='length', directed=True)

    # Step 2: Identify connected components and assign component IDs
    components = list(nx.strongly_connected_components(G_bike))
    component_ids = {}
    for i, component in enumerate(components):
        for node in component:
            component_ids[node] = i

    # Step 3: Build a KDTree for efficient spatial queries
    all_coords = np.vstack([nodes_df['x'], nodes_df['y']]).T
    tree = cKDTree(all_coords)

    # Step 4: Find node pairs within the proximity threshold
    pairs = tree.query_pairs(r=threshold)
    pairs_df = pd.DataFrame(list(pairs), columns=['idx1', 'idx2'])

    # Map node indices to component IDs
    pairs_df['component_id_1'] = nodes_df.iloc[pairs_df['idx1']]['nodeID'].map(component_ids).values
    pairs_df['component_id_2'] = nodes_df.iloc[pairs_df['idx2']]['nodeID'].map(component_ids).values

    # Step 5: Filter pairs connecting different components
    pairs_df = pairs_df[pairs_df['component_id_1'] != pairs_df['component_id_2']]

    # Step 6: Compute Euclidean distances for node pairs
    coords_idx1 = all_coords[pairs_df['idx1']]
    coords_idx2 = all_coords[pairs_df['idx2']]
    pairs_df['distance'] = np.hypot(coords_idx1[:, 0] - coords_idx2[:, 0],
                                    coords_idx1[:, 1] - coords_idx2[:, 1])

    # Step 7: Find the closest pair for each component pair
    pairs_df['component_pair'] = list(zip(pairs_df['component_id_1'], pairs_df['component_id_2']))
    closest_pairs = pairs_df.groupby('component_pair').first().reset_index()

    # Step 8: Add bidirectional edges to connect the components
    new_edges = []
    for _, row in closest_pairs.iterrows():
        idx1, idx2 = row['idx1'], row['idx2']
        node1 = nodes_df.iloc[idx1]
        node2 = nodes_df.iloc[idx2]
        distance = row['distance']

        # Create a two-way edge
        for source, target in [(node1['nodeID'], node2['nodeID']), (node2['nodeID'], node1['nodeID'])]:
            new_edges.append({
                'source': source,
                'target': target,
                'length': distance,
                'speed': 8,  # Default speed for new edges
                'time': (distance / 1000) / 8 * 60,  # Travel time in minutes
                'highway': new_link_label,
                'geometry': LineString([(node1['x'], node1['y']), (node2['x'], node2['y'])])
            })

    # Step 9: Add new edges to the edges DataFrame
    new_edges_df = gpd.GeoDataFrame(new_edges, crs=edges_df.crs)
    updated_edges_df = pd.concat([edges_df, new_edges_df], ignore_index=True)

    # Step 10: Create a new NetworkX graph with updated edges
    G_updated = momepy.gdf_to_nx(updated_edges_df, approach="primal", length='length', directed=True)

    # Step 11: Extract the largest connected subgraph
    largest_component = max(nx.strongly_connected_components(G_updated), key=len)
    largest_subgraph = G_updated.subgraph(largest_component).copy()

    # Step 12: Create updated GeoDataFrames for the largest subgraph
    largest_edges = []
    for u, v, data in largest_subgraph.edges(data=True):
        data['source'] = u
        data['target'] = v
        largest_edges.append(data)
    largest_edges_gdf = gpd.GeoDataFrame(largest_edges, crs=edges_df.crs)
    return largest_edges_gdf


def connect_subgraphs(edges_df, new_link_label, threshold=0.0004):
    # Convert edges_df to a networkx graph
    G_bike = momepy.gdf_to_nx(edges_df, approach="primal", length='length', directed=True)

    def process_batch(components):
        batch_nodes = []
        batch_edges = []
        for component in components:
            if len(component) == 1:
                continue
            else:
                subgraph = G_bike.subgraph(component)
                nodes_data = dict(subgraph.nodes(data=True))
                n_df = pd.DataFrame()
                n_df['nodeID'] = [k for k, v in nodes_data.items()]
                n_df = n_df.join(pd.json_normalize([v for k, v in nodes_data.items()]))
                n_df = gpd.GeoDataFrame(n_df, geometry=gpd.points_from_xy(n_df['x'], n_df['y']), crs="EPSG:4326")
                n_df.set_index('nodeID', inplace=True)
                edges_df = pd.DataFrame(subgraph.edges(data=True))
                edges_df = pd.json_normalize(edges_df[2])  # .drop('mm_len',axis=1)
                batch_nodes.append(n_df)
                batch_edges.append(edges_df)
        return batch_nodes, batch_edges

    # Split components into batches
    batch_size = 100  # Adjust based on your system
    sorted_graph = sorted(nx.strongly_connected_components(G_bike), key=len, reverse=True)
    batches = [sorted_graph[i:i + batch_size] for i in range(0, len(sorted_graph), batch_size)]

    # Process batches in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_batch)(batch) for batch in batches
    )

    # Unpack results
    nodes_gdfs = []
    edges_gdfs = []
    for batch_nodes, batch_edges in results:
        nodes_gdfs.extend(batch_nodes)
        edges_gdfs.extend(batch_edges)

    # Step 1: Assign component IDs and combine all nodes into a single GeoDataFrame
    for idx, nodes_gdf in enumerate(nodes_gdfs):
        nodes_gdf['component_id'] = idx
        nodes_gdf['node_idx'] = nodes_gdf.index  # Preserve node indices within each component

    all_nodes = pd.concat(nodes_gdfs, ignore_index=True)

    # Step 2: Build a KDTree for efficient spatial queries
    all_coords = np.vstack([all_nodes.geometry.x.values, all_nodes.geometry.y.values]).T
    tree = cKDTree(all_coords)

    # Step 3: Find all pairs of nodes within the threshold distance
    pairs = tree.query_pairs(r=threshold)

    # Convert pairs to DataFrame
    pairs_df = pd.DataFrame(list(pairs), columns=['idx1', 'idx2'])

    # Step 4: Get component IDs and filter pairs from different components
    pairs_df['component_id_1'] = all_nodes.iloc[pairs_df['idx1']]['component_id'].values
    pairs_df['component_id_2'] = all_nodes.iloc[pairs_df['idx2']]['component_id'].values

    # Keep only pairs where components are different
    pairs_df = pairs_df[pairs_df['component_id_1'] != pairs_df['component_id_2']]

    # Step 5: Compute distances between node pairs
    coords_idx1 = all_coords[pairs_df['idx1']]
    coords_idx2 = all_coords[pairs_df['idx2']]
    pairs_df['distance'] = np.hypot(coords_idx1[:, 0] - coords_idx2[:, 0],
                                    coords_idx1[:, 1] - coords_idx2[:, 1])

    # Step 6: Keep the closest pair of nodes between each pair of components
    pairs_df['component_pair'] = list(zip(pairs_df['component_id_1'], pairs_df['component_id_2']))
    pairs_df = pairs_df.sort_values('distance')
    min_distances = pairs_df.groupby('component_pair').first().reset_index()

    # Step 7: Prepare Union-Find data structure for Kruskal's algorithm
    parent = {idx: idx for idx in range(len(nodes_gdfs))}

    def find(u):
        # Path compression
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        parent_u = find(u)
        parent_v = find(v)
        if parent_u != parent_v:
            parent[parent_u] = parent_v
            return True  # Merged successfully
        return False  # Already in the same set

    # Step 8: Process edges in order of increasing distance to merge components
    edges_to_add = []
    for _, row in min_distances.iterrows():
        comp_u = int(row['component_id_1'])
        comp_v = int(row['component_id_2'])
        dist = row['distance']

        if dist >= threshold:
            continue  # Skip edges beyond the threshold

        # Attempt to union the components
        if union(comp_u, comp_v):
            # Keep track of the edge to add between components
            idx1 = int(row['idx1'])
            idx2 = int(row['idx2'])
            edges_to_add.append((idx1, idx2, dist))

    # Step 9: Build the new merged components
    # Map old component IDs to new component IDs
    component_map = {idx: find(idx) for idx in parent}

    # Group node indices by their new component IDs
    components = defaultdict(list)
    for idx, comp_id in enumerate(all_nodes['component_id']):
        new_comp_id = component_map[comp_id]
        components[new_comp_id].append(idx)

    # Initialize lists for the new nodes and edges GeoDataFrames
    new_nodes_gdfs = []
    new_edges_gdfs = []

    # Build new components
    for comp_id, node_indices in components.items():
        # Combine nodes
        combined_nodes = all_nodes.iloc[node_indices].copy()
        # Combine edges from original components
        original_comp_ids = set(all_nodes.iloc[node_indices]['component_id'])
        combined_edges = pd.concat([edges_gdfs[old_comp_id] for old_comp_id in original_comp_ids],
                                   ignore_index=True)
        # Add new edges between merged components
        for edge in edges_to_add:
            idx1, idx2, dist = edge
            node1 = all_nodes.iloc[idx1]
            node2 = all_nodes.iloc[idx2]
            if component_map[node1['component_id']] == comp_id:
                # Edge connects to this component
                line = LineString([node1.geometry, node2.geometry])
                new_edge = pd.DataFrame(
                    {'geometry': [line], 'speed': [8], 'edge': [f'M{str(len(new_edges_gdfs)).zfill(5)}'],
                     'data_source': [new_link_label]})
                new_edge = gpd.GeoDataFrame(new_edge, geometry='geometry', crs='EPSG:4326')
                combined_edges = pd.concat([combined_edges, new_edge], ignore_index=True)
        new_nodes_gdfs.append(combined_nodes)
        new_edges_gdfs.append(combined_edges)

    largest_edges_gdf = new_edges_gdfs[0]
    largest_nodes_gdf = new_nodes_gdfs[0]

    largest_edges_gdf = gpd.GeoDataFrame(largest_edges_gdf, crs='epsg:4326')
    largest_edges_gdf['length'] = largest_edges_gdf.to_crs(epsg=3414).length
    largest_edges_gdf['speed'] = np.where(largest_edges_gdf['data_source'] == new_link_label, 8, largest_edges_gdf['speed'])
    largest_edges_gdf['time'] = largest_edges_gdf.apply(lambda x: (x['length'] / 1000) * 1 / x['speed'] * 60, axis=1)
    largest_edges_gdf['edge'] = [f'B{str(x).zfill(5)}' for x in range(len(largest_edges_gdf))]
    largest_nodes_gdf = largest_nodes_gdf.rename(columns={"node_idx": "nodeID"})

    return largest_edges_gdf, largest_nodes_gdf

def calculate_distance_matrix(nodes_gdfs):
    # This runs as part of the connect_graph function
    # Basically, the distance between every disconnected subgraph is calculated.
    # This is done by calculating the pairwise distance between all nodes in both subgraphs and taking the smallest distance
    size = len(nodes_gdfs)
    matrix = np.full((size, size), np.inf)

    for i, gdf_init in enumerate(nodes_gdfs):
        known_xy = np.stack([gdf_init.geometry.x, gdf_init.geometry.y], -1)
        tree = scipy.spatial.cKDTree(known_xy)

        for j, gdf_pair in enumerate(nodes_gdfs):
            if i != j and matrix[i, j] == np.inf:
                query_xy = np.stack([gdf_pair.geometry.x, gdf_pair.geometry.y], -1)
                distances, indices = tree.query(query_xy, k=1)
                min_dist = distances.min()
                matrix[i, j] = min_dist
                matrix[j, i] = min_dist

    return matrix

def connect_graph(edges_df, new_link_label, threshold=0.0004):
    # Threshold is maximum length of link that can be introduced to connect two subgraphs
    # Threshold unit is in degrees (input dataframes must be in epsg=4326)
    # If the smallest distance between two graphs (calculated in the calculate_distance_matrix function)
    # is below the threshold, we identify the two nodes on both subgraphs that are closest to each other
    # and manually add a link between the subgraphs.
    graph = momepy.gdf_to_nx(edges_df, approach="primal", length='length')
    sorted_graph = sorted(nx.connected_components(graph), key = len, reverse=True)
    largest_connected = graph.copy()

    nodes_gdfs = []
    edges_gdfs = []

    # for every disconnected component of the graph
    for i in range(len(sorted_graph)):
        a = largest_connected.copy()
        a.remove_nodes_from([n for n in a if n not in set(sorted_graph[i])])
        nodes, edges = momepy.nx_to_gdf(a)
        nodes_gdfs.append(nodes)
        edges_gdfs.append(edges)

    matrix = calculate_distance_matrix(nodes_gdfs)

    matrix_backup = matrix.copy()
    nodes_gdfs_backup = nodes_gdfs.copy()
    edges_gdfs_backup = edges_gdfs.copy()

    matrix = matrix_backup.copy()
    nodes_gdfs = nodes_gdfs_backup.copy()
    edges_gdfs = edges_gdfs_backup.copy()

    x = 1

    while matrix.min() < threshold: #20m in geographic projection
        # Find the pair of components with the minimum distance
        i, j = np.unravel_index(np.argmin(matrix), matrix.shape)

        # Get the closest nodes
        known_xy = np.stack([nodes_gdfs[i].geometry.x, nodes_gdfs[i].geometry.y], -1)
        tree = scipy.spatial.cKDTree(known_xy)

        query_xy = np.stack([nodes_gdfs[j].geometry.x, nodes_gdfs[j].geometry.y], -1)
        distances, indices = tree.query(query_xy, k=1)

        idx1 = indices[distances.argmin()]
        idx2 = distances.argmin()

        node1 = nodes_gdfs[i].iloc[idx1]
        node2 = nodes_gdfs[j].iloc[idx2]

        # Combine the two components
        combined_edges = pd.concat([edges_gdfs[i], edges_gdfs[j]])
        new_edge = LineString([node1.geometry, node2.geometry])
        new_edge_gdf = gpd.GeoDataFrame({'geometry': [new_edge], 'speed': [8], 'edge': [f'M{str(x).zfill(5)}'],
            'data_source': [new_link_label]})
        new_edge_gdf = new_edge_gdf.set_crs(epsg=4326)
        combined_edges = pd.concat([combined_edges, new_edge_gdf])

        # Remove the combined components from the lists
        for idx in sorted([i, j], reverse=True):
            del nodes_gdfs[idx]
            del edges_gdfs[idx]

        # Create the new combined component
        combined_g = momepy.gdf_to_nx(combined_edges, approach="primal", length='length')
        combined_nodes, combined_edges = momepy.nx_to_gdf(combined_g)

        # Add the new combined component back to the lists
        nodes_gdfs.append(combined_nodes)
        edges_gdfs.append(combined_edges)

        # Update the distance matrix
        matrix = np.delete(matrix, [i, j], axis=0)
        matrix = np.delete(matrix, [i, j], axis=1)

        new_distances = []
        known_xy = np.stack([combined_nodes.geometry.x, combined_nodes.geometry.y], -1)
        tree = scipy.spatial.cKDTree(known_xy)

        for gdf in nodes_gdfs[:-1]:  # Exclude the last one (new combined component)
            query_xy = np.stack([gdf.geometry.x, gdf.geometry.y], -1)
            distances, indices = tree.query(query_xy, k=1)
            min_dist = distances.min()
            new_distances.append(min_dist)

        new_distances = np.array(new_distances)
        new_distances_with_inf = np.append(new_distances, [np.inf])
        matrix = np.c_[matrix, new_distances]
        matrix = np.r_[matrix, [new_distances_with_inf]]

        x+=1

    gdf = edges_gdfs[np.argmax([gdf.shape[0] for gdf in edges_gdfs])]

    if 'source' in gdf.columns:
        gdf = gdf.drop(["source"], axis=1)
    if 'target' in gdf.columns:
        gdf = gdf.drop(["target"], axis=1)

    G_connected = momepy.gdf_to_nx(gdf, directed=True)
    node_df, _ = momepy.nx_to_gdf(G_connected)
    mapping = {(row['x'], row['y']): row["nodeID"] for _, row in node_df.iterrows()}
    G_connected = nx.relabel_nodes(G_connected, mapping)
    node_df, graph_df = momepy.nx_to_gdf(G_connected)
    graph_df = graph_df.drop(['mm_len'], axis=1).rename(columns={'node_start':'source', 'node_end':'target'})

    graph_df_ = graph_df.to_crs(epsg=3414)
    graph_df['length'] = graph_df_.length
    graph_df['time'] = ((graph_df['length']/1000) * 1/graph_df['speed'])*60

    return graph_df, node_df