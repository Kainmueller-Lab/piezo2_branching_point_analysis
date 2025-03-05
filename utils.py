import numpy as np
import networkx as nx
import napari


def replace(array, old_values, new_values):
    """fast function to replace set of values in array with new values"""
    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values

    return values_map[array]


def filter_components(volume, thresh):
    """remove instances smaller than `thresh` pixels"""
    labels, counts = np.unique(volume, return_counts=True)
    small_labels = labels[counts <= thresh]

    volume = replace(
        volume,
        np.array(small_labels),
        np.array([0] * len(small_labels))
    )
    return volume


def update_segment(graph, old_segment_ids, new_segment_pos):
    # update one segment (from one root/branching point to another branching
    # point/ end point with resampled points
    npoints = len(graph.nodes)
    succ = old_segment_ids[0]
    cid = np.max(graph.nodes) + 1
    # insert new nodes and edges
    for pos in new_segment_pos:
        graph.add_node(cid, coord=pos)
        graph.add_edge(succ, cid)
        succ = cid
        cid += 1
    # add last edge to original segment end point
    graph.add_edge(succ, old_segment_ids[-1])

    # remove nodes and edges from resampled segment
    for cid in old_segment_ids[1:-1]:
        graph.remove_node(cid)
    if len(new_segment_pos) > 0:
        if graph.has_edge(old_segment_ids[0], old_segment_ids[-1]):
            graph.remove_edge(old_segment_ids[0], old_segment_ids[-1])

    return graph


# adapted from https://rdrr.io/cran/nat/src/R/neuron.R#sym-resample_segment
def resample_segment(seg, stepsize):
    if seg.shape[0] < 2:
        return None
    seg_xyz = seg[:, :3]
    l = np.sum(np.sqrt(np.sum(np.diff(seg_xyz, axis=0) ** 2, axis=1)))
    if l <= stepsize:
        return None

    internal_points = np.arange(stepsize, l, stepsize)
    # if last resample point is close to end point of segment, discard it
    if abs(internal_points[-1] - l) < stepsize:
        internal_points = internal_points[:-1]
        if len(internal_points) == 0:
            return None

    cumlength = np.insert(np.cumsum(np.sqrt(
        np.sum(np.diff(seg_xyz, axis=0) ** 2, axis=1))), 0, 0)

    seg_new = np.empty((len(internal_points), seg.shape[1]))
    for i in range(seg.shape[1]):
        if np.all(np.isfinite(seg[:, i])):
            seg_new[:, i] = np.interp(internal_points, cumlength, seg[:, i])
        else:
            seg_new[:, i] = np.nan
    return seg_new


def resample_graph(graph, stepsize):
    graph_resampled = graph.copy()
    # get roots
    roots = [n for n, d in graph.in_degree() if d==0]
    positions = nx.get_node_attributes(graph, 'coord')

    # traverse tree
    for root in roots:
        start = root
        next_starts = []
        while graph.out_degree(start) > 0:
            childs = list(graph.successors(start))
            for child in childs:
                # find complete segment up to the next branching point
                segment_points = [start]
                segment_pos = [positions[start]]
                cnode = child
                while graph.out_degree(cnode) == 1:
                    segment_points.append(cnode)
                    segment_pos.append(positions[cnode])
                    cnode = list(graph.successors(cnode))[0]
                segment_points.append(cnode)
                segment_pos.append(positions[cnode])
                # resample segment
                segment_resampled = resample_segment(
                    np.array(segment_pos), stepsize)
                # add to resampled graph
                if segment_resampled is not None:
                    graph_resampled = update_segment(
                        graph_resampled, segment_points, segment_resampled)
                if segment_resampled is None and len(segment_pos) > 2:
                    graph_resampled = update_segment(
                            graph_resampled, segment_points, [])

                # if segment end point is branching point, add it as next start
                if graph.out_degree(cnode) > 1:
                    next_starts.append(cnode)
            if len(next_starts) > 0:
                start = next_starts.pop()
            else:
                break

    return graph_resampled


def convert_to_ordered_tree(G, root):
    """
    Converts an unordered NetworkX graph into a directed ordered tree given a root node.

    Parameters:
    G (networkx.Graph): The input undirected graph.
    root (node): The root node for the ordered tree.

    Returns:
    networkx.DiGraph: A directed version of the tree where edges point away from the root.
    """
    # Initialize directed graph
    ordered_tree = nx.DiGraph()

    # Perform BFS to establish a hierarchy
    root = int(root)
    visited = set()
    queue = [root]
    ordered_tree.add_node(root)
    ordered_tree.nodes[root]["coord"] = G.nodes[root]["coord"]

    while queue:
        parent = queue.pop(0)  # BFS: process nodes in order
        visited.add(parent)

        for child in G.neighbors(parent):
            child = int(child)
            if child not in visited:
                ordered_tree.add_node(child)
                ordered_tree.nodes[child]["coord"] = G.nodes[child]["coord"]
                ordered_tree.add_edge(parent, child)  # Direct the edge away from root
                queue.append(child)

    return ordered_tree


def unit_vector(vector):
    """ Returns the unit vector of a given vector. """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle (in degrees) between two vectors using the dot product. """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def get_branching_point_angles(tree, root, branching_point_ids, image=None):
    # convert tree to ordered tree
    di_tree = convert_to_ordered_tree(tree, root)
    # resample tree to have higher distance to branching point when measuring angle
    di_tree_res = resample_graph(di_tree, 20)
    angles = {}

    for i, node in enumerate(branching_point_ids):
        if node not in di_tree_res.nodes:
            continue  # Skip if node is not in the graph

        successors = list(di_tree_res.successors(node))
        predecessors = list(di_tree_res.predecessors(node))

        if len(successors) == 2:  # Ensure it's a branching point
            pos = np.array(di_tree_res.nodes[node]['coord'])
            pos_out1 = np.array(di_tree_res.nodes[successors[0]]['coord'])
            pos_out2 = np.array(di_tree_res.nodes[successors[1]]['coord'])

            v_out1 = pos_out1 - pos
            v_out2 = pos_out2 - pos

            angle_out = angle_between(v_out1, v_out2)  # Between outgoing edges

            if len(predecessors) == 1:  # If there is a single incoming edge
                pos_in = np.array(di_tree_res.nodes[predecessors[0]]['coord'])
                v_in = pos - pos_in

                angle_in_out1 = angle_between(v_in, v_out1)
                angle_in_out2 = angle_between(v_in, v_out2)
            else:
                angle_in_out1 = angle_in_out2 = None  # No single incoming edge

            angles[node] = (angle_out, angle_in_out1, angle_in_out2)
            print("b%i" % i, node, angle_out, angle_in_out1, angle_in_out2)

    #if image is not None:
    #    plot_angles_on_image(image, di_tree_res, branching_point_ids, angles)

    return angles


def plot_angles_on_image(image_volume, ordered_tree, branching_point_ids, angles_dict):
    viewer = napari.Viewer()
    viewer.add_image(image_volume, colormap='gray', blending='additive')

    # Extract node positions
    node_positions = np.array([ordered_tree.nodes[n]['coord'] for n in ordered_tree.nodes])
    node_colors = ['white' if n not in branching_point_ids else 'red' for n in ordered_tree.nodes]

    # Add nodes
    viewer.add_points(node_positions, size=5, face_color=node_colors, name="Graph Nodes")

    # Add edges
    edge_lines = []
    for u, v in ordered_tree.edges:
        edge_lines.append([ordered_tree.nodes[u]['coord'], ordered_tree.nodes[v]['coord']])
    viewer.add_shapes(edge_lines, shape_type='line', edge_color='yellow', name="Graph Edges")

    # Add branching angles as text labels
    text_positions = []
    text_values = []

    for node in branching_point_ids:
        if node in angles_dict:
            x, y, z = ordered_tree.nodes[node]['coord']
            angle_out, angle_in_out1, angle_in_out2 = angles_dict[node]
            text_positions.append([x, y, z])
            text_values.append(f"{angle_out:.1f}°\n{angle_in_out1:.1f}°\n{angle_in_out2:.1f}°")
    text = {
            'string': '{label}',
            'size': 16,
            'color': 'white',
            'translation': np.array([-10, -10, -10]),
        }

    if text_positions:
        viewer.add_points(text_positions, size=1, text=text, face_color='none', name="Angles", properties={"label": text_values})
    
    napari.run()


def count_trifurcations(tree, branching_points_sorted, dists, root, distance_threshold):
    num_tris = 0
    paths2root = [nx.shortest_path(tree, root, b) for b in branching_points_sorted]
    for i, b_id in enumerate(branching_points_sorted):
        # skip first branching point, as we compare distance to parent branching point
        if i == 0:
            continue
        b_id_dist = dists[i] 
        # get parent
        path2root = paths2root[i]
        # path2root is list from root to current branching point
        path2root = np.flip(path2root[:-1])
        p_id = -1
        for p in path2root:
            if p in branching_points_sorted:
                p_id = p
                break
        p_id_dist = dists[branching_points_sorted == p_id]
        if b_id_dist - p_id_dist < distance_threshold:
            num_tris += 1
    return num_tris

