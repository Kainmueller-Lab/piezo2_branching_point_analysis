import os
from glob import glob
import numpy as np
import h5py
from skimage import measure, feature, segmentation
from skimage import filters, morphology
from skimage import io
from scipy import ndimage
from scipy.spatial import KDTree
import tifffile
import time
import json
from skimage.morphology import ball
import networkx as nx
import kimimaro
from cloudvolume import Skeleton
import argparse
from PIL import Image, ImageDraw, ImageFont
from imaris_ims_file_reader.ims import ims
from utils import count_trifurcations, get_branching_point_angles
from manual_roots import manual_roots_right, manual_roots_left 


def get_arguments():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--in_folder", type=str
            )
    parser.add_argument(
            "--in_file", type=str
            )
    parser.add_argument(
            "--in_key", type=str, default=None
            )
    parser.add_argument(
            "--mask_cleaned", type=str, default=None
            )
    parser.add_argument(
            "--out_folder", type=str,
            default="/home/maisl/workspace/piezo2/results"
            )
    parser.add_argument(
            "--max_radius", type=int, default=8
            )
    parser.add_argument(
            "--dilation_factor", type=int, default=7
            )
    parser.add_argument(
            "--skel_scale", type=int, default=5
            )
    parser.add_argument(
            "--manual_root", nargs="+", type=float, default=None
            )
    parser.add_argument(
        "--only_get_tri", action="store_true", default=False,
        help="flag to only compute number of trifurcations"
    )
    parser.add_argument(
        "--tri_thresh", type=int, default=30,
        help="distance threshold to count trifurcation in microns"
    )

    args = parser.parse_args()
    return args


def analyze_tree(tree, outfn, mip_raw_xy=None, mip_raw_yz=None, mask=None,
        resolution=None, root=None, tree_label=None,
        manual_root=None, only_get_tri=False, tri_thresh=30):
    start_time = time.time()
    sample_id = os.path.basename(outfn).split("_")[0]
    # get skeleton points
    coords_skel = nx.get_node_attributes(tree, "coord")
    coords_skel = np.array([v for k, v in coords_skel.items()])
    
    # get branching points
    lbl, cnt = np.unique(tree.edges, return_counts=True)
    branch_idx = lbl[cnt > 2]
    end_idx = lbl[cnt == 1]
    coords_branch = np.array([tree.nodes[k]["coord"] for k in branch_idx])
    radii = np.array([tree.nodes[k]["radius"] for k in end_idx])

    # get root: end point with highest radius
    if root is None:
        root = end_idx[np.argmax(radii)]
    coord_root = tree.nodes[root]["coord"] 
    path2root = [nx.shortest_path(tree, root, b) for b in branch_idx]
    
    # adapted cable length method from cloud-volume here for distance in physical space
    dist2root_microns = []
    for path in path2root:
        path_coords = np.array([tree.nodes[p]["coord"] for p in path])
        path_coords = path_coords * np.array(
                [resolution[2], resolution[1], resolution[0]])
        path_diff = path_coords[:-1] - path_coords[1:]
        path_diff *= path_diff
        dist = np.sum(path_diff, axis=1)
        dist = np.sum(np.sqrt(dist))
        dist2root_microns.append(dist)
    
    dist2root_microns = np.array(dist2root_microns)
    radii_branch = np.array([tree.nodes[k]["radius"] for k in branch_idx])
    # todo: maybe get radius of child instead?
    sort_idx = np.argsort(dist2root_microns)
    branch_idx_sorted = branch_idx[sort_idx]
    coords_branch_sorted = coords_branch[sort_idx]
    dist2root_microns = dist2root_microns[sort_idx]
    radii_branch = radii_branch[sort_idx]
    
    # count number of "trifurcations"
    num_tris = count_trifurcations(tree, branch_idx_sorted, dist2root_microns, root, tri_thresh)
    print("%i trifurcations for tree %i" % (num_tris, tree_label))
    
    # get branching point angles
    image_for_angles = mask.astype(np.uint8)
    image_for_angles = ((image_for_angles / float(np.max(image_for_angles))) * 255).astype(np.uint8)
    angles = get_branching_point_angles(tree, root, branch_idx_sorted, image=image_for_angles)
    
    if only_get_tri:
        return None, None, None

    # visualize in array
    skel_array = np.zeros_like(mask).astype(np.uint8)
    branch_array = np.zeros_like(mask).astype(np.uint8)
    skel_array[tuple(coords_skel.astype(int).T)] = 1 #clabel
    branch_array[tuple(coords_branch.astype(int).T)] = 1 #clabel
    # dilate root in skeleton array
    rz = int(coord_root[0])
    ry = int(coord_root[1])
    rx = int(coord_root[2])
    skel_array[rz-3:rz+3, ry-3:ry+3, rx-3:rx+3] = 1
    #skel_array = morphology.dilation(skel_array)
    branch_array = morphology.dilation(branch_array, footprint=ball(5))
    
    mip_skel_xy = (np.max(skel_array, axis=0) * 255).astype(np.uint8)
    mip_branch_xy = (np.max(branch_array, axis=0) * 255).astype(np.uint8)
    mip_tree_xy = np.stack([mip_raw_xy, mip_skel_xy, mip_branch_xy], axis=-1)
    mip_mask_xy = (np.max(mask, axis=0) * 255).astype(np.uint8)
    mip_tree_xy_mask = np.stack([mip_mask_xy, mip_skel_xy, mip_branch_xy], axis=-1)
    
    mip_skel_yz = (np.max(skel_array, axis=2) * 255).astype(np.uint8)
    mip_branch_yz = (np.max(branch_array, axis=2) * 255).astype(np.uint8)
    mip_tree_yz = np.stack([mip_raw_yz, mip_skel_yz, mip_branch_yz], axis=-1)
    mip_tree_yz = np.moveaxis(mip_tree_yz, 0, 1)
    mip_mask_yz = (np.max(mask, axis=2) * 255).astype(np.uint8)
    mip_tree_yz_mask = np.stack([mip_mask_yz, mip_skel_yz, mip_branch_yz], axis=-1)
    mip_tree_yz_mask = np.moveaxis(mip_tree_yz_mask, 0, 1)
    
    # define pillow images
    mip_xy = Image.fromarray(mip_tree_xy)
    mip_text_xy = ImageDraw.Draw(mip_xy)
    mip_yz = Image.fromarray(mip_tree_yz)
    mip_text_yz = ImageDraw.Draw(mip_yz)
    mask_xy = Image.fromarray(mip_tree_xy_mask)
    mask_text_xy = ImageDraw.Draw(mask_xy)
    mask_yz = Image.fromarray(mip_tree_yz_mask)
    mask_text_yz = ImageDraw.Draw(mask_yz)
    myfont = ImageFont.truetype("FreeMonoBold.ttf", 18)
    
    lines = []
    #lines.append("sample_id, branch point name, branch point id, distance in microns")
    last_y = 0
    for i, c_branch in enumerate(branch_idx_sorted):
        c_dist = dist2root_microns[i]
        #c_radius = radii_branch[i]
        c_coord = coords_branch_sorted[i]
        bname = "b%i" % (i)
        btext = bname + ": %.2f" % c_dist
        y = c_coord[1]
        if abs(y-last_y) < 10:
            if (y-last_y) < 0:
                y_shift = -20
            else:
                y_shift = 0
        else:
            y_shift = - 10
        last_y = y
        #legend_pos = max(0, mask_xy.height - (5 + (len(branch_idx_sorted) - i) * 10))
        legend_pos = min(mask_xy.height, 5 + i * 20)
        mip_text_xy.text((c_coord[2] + 5, y + y_shift), bname, font=myfont,
                fill=(255, 255, 255))
        mip_text_yz.text((c_coord[0] + 5, y + y_shift), bname, font=myfont,
                fill=(255, 255, 255))
        mip_text_xy.text((5, legend_pos), btext, font=myfont, fill=(255, 255, 255))
        
        mask_text_xy.text((c_coord[2] + 5, y + y_shift), bname, font=myfont,
                fill=(255, 255, 255))
        mask_text_yz.text((c_coord[0] + 5, y + y_shift), bname, font=myfont,
                fill=(255, 255, 255))
        mask_text_xy.text((5, legend_pos), btext, font=myfont, fill=(255, 255, 255))
        x,y,z = np.flip(c_coord)
        lines.append("%s,%i,%s,%i,%i,%i,%i,%f" % (sample_id, tree_label, bname, c_branch, x, y, z, c_dist))
    
    # name root
    if tree_label is None:
        root_name = "root"
    else:
        root_name = "r%i" % int(tree_label)
    mip_text_xy.text((rx + 5, ry - 5), root_name, font=myfont,
            fill=(255, 255, 255))
    mip_text_yz.text((rz + 5, ry - 5), root_name, font=myfont,
            fill=(255, 255, 255))
    mask_text_xy.text((rx + 5, ry - 5), root_name, font=myfont,
            fill=(255, 255, 255))
    mask_text_yz.text((rz + 5, ry - 5), root_name, font=myfont,
            fill=(255, 255, 255))
    
    mip1 = Image.new('RGB', (mip_xy.width + mip_yz.width, mip_xy.height))
    mip2 = Image.new('RGB', (mask_xy.width + mask_yz.width, mask_xy.height))
    mip1.paste(mip_xy, (0, 0))
    mip1.paste(mip_yz, (mip_xy.width, 0))
    mip2.paste(mask_xy, (0, 0))
    mip2.paste(mask_yz, (mask_xy.width, 0))
    
    # save annotated mip
    mip1.save(outfn + "_raw.png")
    mip2.save(outfn + "_mask.png")

    # save csv
    with open(outfn + ".csv", "w") as outf:
        outf.write(
                "sample_id, tree, branch point name, branch point id, x, y, z, distance in microns\n")
        root_line = "%s,%i,%s,%i,%i,%i,%i,%f" % (sample_id, tree_label, root_name, int(root), 
                coord_root[2], coord_root[1], coord_root[0], 0.0)
        outf.write(root_line + "\n")
        for line in lines:
            outf.write(line + "\n")

    # save tree to json
    result_dict = {
            "root": int(root),
            "branch_points": [int(n) for n in branch_idx_sorted],
            "branch_dist": [float(d) for d in dist2root_microns],
            "branch_radius": [float(d) for d in radii_branch],
            "nodes": [int(n) for n in tree.nodes],
            "coords": [[int(x), int(y), int(z)] for x, y, z in coords_skel]
            }
    if manual_root is not None:
        result_dict["manual_root"] = [int(p) for p in manual_root]
    else:
        result_dict["manual_root"] = None
    with open(outfn + ".json", "w") as outf:
        json.dump(result_dict, outf)
    print("--- %s seconds for analyzing tree ---" % (time.time() - start_time))

    return coord_root, coords_skel, coords_branch


def skeletonize_mask(mask, out_folder="", sample_name="",
        dilation_factor=11, max_radius=8, raw=None, resolution=None,
        manual_root=None, skel_scale=5, only_get_tri=False, tri_thresh=30):
    
    # binary closing
    start_time = time.time()
    mask_fn = os.path.join(out_folder, "%s_mask_closed_dil_%i.tiff" % (
        sample_name, dilation_factor))
    if os.path.exists(mask_fn):
        mask_closed = io.imread(mask_fn)
        mask_closed = mask_closed > 0
        #print("mask loaded: ", mask_closed.shape, mask_closed.dtype, np.sum(mask_closed > 0))
    else:
        mask_closed = morphology.binary_closing(mask, footprint=ball(dilation_factor))
        io.imsave(mask_fn, mask_closed, check_contrast=False)
        mip = (np.max(mask_closed, axis=0) * 255).astype(np.uint8)
        io.imsave(mask_fn.replace(".tiff",".png"), mip, check_contrast=False)
    print("--- %s seconds for binary closing ---" % (time.time() - start_time))

    # get connected components of dilated mask
    mask_closed_labeled, _ = ndimage.label(mask_closed)

    # skeletonize with kimimaro
    start_time = time.time()
    skel_fn = os.path.join(out_folder, "%s_skel" % sample_name)
    if os.path.exists(skel_fn + "_1.swc"):
        skeleton = {}
        inf =  open(skel_fn + "_1.swc", "r")
        swc_data = inf.read()
        skeleton[1] = Skeleton().from_swc(swc_data)
    else:
        skeleton = kimimaro.skeletonize(
                mask_closed,
                teasar_params={
                    "scale": skel_scale,
                    "const": skel_scale,
                    "pdrf_scale": 100000,
                    "pdrf_exponent": 4,
                    "soma_acceptance_threshold": 9, # physical units
                    "soma_detection_threshold": 9, # physical units
                    "soma_invalidation_const": 4, # physical units
                    "soma_invalidation_scale": 4, # physical units
                    "max_paths": 1000, # default None
                    },
                extra_targets_before=manual_root,
                dust_threshold=0, # skip connected components with fewer than this many voxels
                anisotropy=(1,1,1), # default True
                fix_branching=True, # default True
                fix_borders=True, # default True
                fill_holes=True, # default False
                fix_avocados=False, # default False
                progress=True, # default False, show progress bar
                parallel=1, # <= 0 all cpu, 1 single process, 2+ multiprocess
                parallel_chunk_size=100, # how many skeletons to process before updating progress bar
            )
        # save all skeletons to swc
        for k in skeleton.keys():
            skel = skeleton[k]
            outfn = skel_fn + "_%i.swc" % k
            with open(outfn, "w") as fout:
                fout.write(skel.to_swc())
    print("--- %s seconds for skeletonization ---" % (time.time() - start_time))
    
    # smooth skeleton and visualize
    skeleton_array = np.zeros_like(mask_closed).astype(np.uint8)
    for k in skeleton.keys():
        skeleton[k] = skeleton[k].average_smoothing(5, check_boundary=False)
        coords_skel = skeleton[k].vertices
        skeleton_array[tuple(coords_skel.astype(int).T)] = 1

    prefix = "%s_dil_%i_rad_%i" % (sample_name, dilation_factor, max_radius)
    io.imsave(out_folder + "/" + prefix + "_skeleton_complete.tiff", 
            (skeleton_array * 255).astype(np.uint8), check_contrast=False)
    #raw_fn = out_folder + "/%s_raw.tiff" % sample_name
    #io.imsave(raw_fn, raw)
    mip = (np.max(skeleton_array, axis=0) * 255).astype(np.uint8)
    if manual_root is not None:
        mip = np.stack([mip, mip, mip], axis=-1)
        if len(manual_root.shape) == 2:
            for mroot in manual_root:
                mip[mroot[1], mroot[2]] = [255, 0, 0]
        else:
            mip[manual_root[1], manual_root[2]] = [255, 0, 0]
    io.imsave(out_folder + "/" + prefix + "_skeleton_complete.png", mip, check_contrast=False)
    
    start_time = time.time()
    root_ids = []
    root_idx = None
    if manual_root is not None:
        num_trees = 0
        graphs = []
        if len(manual_root.shape) == 1:
            manual_root = [manual_root]
        for mroot in manual_root:
            graph = nx.Graph()
            # get closest point on skeleton
            if len(list(skeleton.keys())) > 1:
                raise NotImplementedError
            for k in skeleton.keys():
                skel = skeleton[k]
                tree = KDTree(skel.vertices)
                dist, idx = tree.query(mroot, k=1)
                root_idx = idx
                root_ids.append(root_idx)
                # get neighboring edges
                num_edges = np.sum(skel.edges[:, 0] == root_idx) + np.sum(
                        skel.edges[:, 1] == root_idx)
                #if num_edges > 2 or num_edges == 0:
                #    raise NotImplementedError
                # create networkx graph
                graph.add_nodes_from(range(len(skel.vertices)))
                graph.add_edges_from(skel.edges)

                # remove all connected components that don't include the root
                for cc in nx.connected_components(graph):
                    subgraph = graph.subgraph(cc).copy()
                    if root_idx in subgraph.nodes:
                        graph = subgraph
                        break
                for n in graph.nodes:
                    graph.nodes[n]["coord"] = skel.vertices[n]
                    graph.nodes[n]["radius"] = skel.radius[n]
            if num_edges == 2:
                # decide which part with higher min y value
                neighbors = np.concatenate([
                    skel.edges[skel.edges[:,0] == root_idx, 1],
                    skel.edges[skel.edges[:,1] == root_idx, 0]])

                graph_a = graph.copy()
                graph_a.remove_edge(neighbors[0], root_idx)
                for cc in nx.connected_components(graph_a):
                    subgraph = graph_a.subgraph(cc).copy()
                    if root_idx in subgraph.nodes:
                        graph_a = subgraph.copy()
                        break
                graph_b = graph.copy()
                graph_b.remove_edge(neighbors[1], root_idx)
                for cc in nx.connected_components(graph_b):
                    subgraph = graph_b.subgraph(cc).copy()
                    if root_idx in subgraph.nodes:
                        graph_b = subgraph.copy()
                        break
                coords_a = nx.get_node_attributes(graph_a, "coord")
                coords_a = np.array([v for k, v in coords_a.items()])
                coords_b = nx.get_node_attributes(graph_b, "coord")
                coords_b = np.array([v for k, v in coords_b.items()])
                if np.max(coords_a[:, 1]) < np.max(coords_b[:, 1]):
                    graph = graph_b.copy()
                else:
                    graph = graph_a.copy()
            
            # prune short end point branches
            #lbl, cnt = np.unique(graph.edges, return_counts=True)
            #end_idx = lbl[cnt == 1]
            graphs.append(graph.copy())
            num_trees += 1
    else:
        # convert kimimaro skeleton to networkx graph
        # data structure: skel.vertices, skel.edges, skel.radius
        for k in skeleton.keys():
            skel = skeleton[k]
            # filter out points with radius larger than 8
            to_keep = skel.radius < max_radius
            # iterate through edges and create graph
            for e1, e2 in skel.edges:
                if to_keep[e1]:
                    if not (e1 in graph.nodes):
                        graph.add_node(e1)
                        graph.nodes[e1]["coord"] = skel.vertices[e1]
                        graph.nodes[e1]["radius"] = skel.radius[e1]
                if to_keep[e2]:
                    if not (e2 in graph.nodes):
                        graph.add_node(e2)
                        graph.nodes[e2]["coord"] = skel.vertices[e2]
                        graph.nodes[e2]["radius"] = skel.radius[e2]
                if to_keep[e1] and to_keep[e2]:
                    if not ((e1, e2) in graph.edges):
                        graph.add_edge(e1, e2)
        
        # remove small subgraphs
        len_subgraphs = [len(c) for c in sorted(nx.connected_components(graph),
            key=len, reverse=True)]
        subgraphs = [c for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
        num_trees = 3
        subgraphs = subgraphs[:num_trees]
        graphs = [graph.subgraph(subgraphs[i]).copy() for i in range(len(subgraphs))]
    #print("--- %s seconds for networkx conversion ---" % (time.time() - start_time))

    # visualize in array
    skeleton_array = np.zeros_like(mask_closed).astype(np.uint8)
    branch_points_array = np.zeros_like(mask_closed).astype(np.uint8)
    roots_array = np.zeros_like(mask_closed).astype(np.uint8)
    mip_raw_xy = np.max(raw, axis=0).astype(np.uint8)
    mip_raw_yz = np.max(raw, axis=2).astype(np.uint8)
    
    # analyze networkx skeletons
    #branch_id_offset = 0
    for i in range(num_trees):
        clabel = i + 1
        root_idx = root_ids[i]
        if manual_root is not None:
            mroot = manual_root[i]
        else:
            mroot = None
        outfn = os.path.join(out_folder, prefix + "_tree_%i" % clabel)
        ctree = graphs[i]
        # get connected component
        cc_lbl = mask_closed_labeled[tuple(mroot)]
        cc_mask = np.logical_and(mask, mask_closed_labeled == cc_lbl)
        coord_root, coords_skel, coords_branch = analyze_tree(
                ctree, outfn,
                mip_raw_xy=mip_raw_xy,
                mip_raw_yz=mip_raw_yz,
                mask=cc_mask,
                resolution=resolution,
                root=root_idx,
                tree_label = clabel,
                manual_root = mroot,
                only_get_tri = only_get_tri,
                tri_thresh = tri_thresh
                )
        if not only_get_tri:
            skeleton_array[tuple(coords_skel.astype(int).T)] = 1 #clabel
            branch_points_array[tuple(coords_branch.astype(int).T)] = 1 #clabel
            roots_array[tuple(coord_root.astype(int).T)] = 1
    
    if not only_get_tri:
        skeleton_array = morphology.dilation(skeleton_array)
        branch_points_array = morphology.dilation(branch_points_array, footprint=np.ones((5,5,5)))
        roots_array = morphology.dilation(roots_array, footprint=np.ones((5,5,5))).astype(bool)
        skeleton_array = (skeleton_array * 255).astype(np.uint8)
        branch_points_array = (branch_points_array * 255).astype(np.uint8)
        result = np.stack([raw, skeleton_array, branch_points_array], axis=-1).astype(np.uint8)
        result[roots_array] = [255, 255, 255]
        result = np.moveaxis(result, -1, 1)
        result_fn = out_folder + "/" + prefix + "_result.tiff"
        # add resolution to tiff file
        # e.g. Voxel size: 3.8927x3.8953x8.0132 micron^3
        tifffile.imwrite(result_fn, result, imagej=True,
                resolution=(1./resolution[2], 1./resolution[1]),
                compression="zlib",
                metadata={
                    "spacing": resolution[0],
                    "unit": "um",
                    "axes": "ZCYX",
                    "mode": "composite"
                    }
                )
        mip_skel = np.max(skeleton_array, axis=0)
        mip_branch = np.max(branch_points_array, axis=0)
        mip_result = np.stack([mip_raw_xy, mip_skel, mip_branch], axis=-1)
        io.imsave(out_folder + "/" + prefix + "_mip_result.png", mip_result, check_contrast=False)


def main():
    args = get_arguments()
    print(args)
    if args.in_folder is None and args.in_file is None:
        print("Please define input folder or input file.")
        exit()
    if args.in_file is not None:
        infns = [args.in_file]
    else:
        infns = glob(args.in_folder + "/*.ims")
            
    for infn in infns:
        sample_name = os.path.basename(infn).split(".")[0]
        sample_id = sample_name.split("_")[0]
        sample_name = sample_id
        print(sample_name, sample_id)
        out_fn = "%s_dil_%i_skel_%i" % (sample_name, args.dilation_factor,
                args.skel_scale)
        out_folder = os.path.join(args.out_folder, out_fn)
        os.makedirs(out_folder, exist_ok=True)
        
        # read raw and mask image
        if infn.endswith(".ims"):
            # check out imaris reader
            ds = ims(infn, ResolutionLevelLock=2, resolution_decimal_places=4)
            resolution = ds.resolution
            # read in raw
            raw = np.array(ds[2,0,0,:,:,:])
            raw = raw.astype(np.float32)
            perc = np.percentile(raw, 99.9)
            raw = np.clip(raw, 0, perc)
            raw = raw / perc
            raw = (raw * 255).astype(np.uint8)
            mask = np.array(ds[2,0,1,:,:,:])
            ds.close()
        elif infn.endswith(".tiff"):
            in_image = tifffile.imread(infn)
            in_image = np.moveaxis(in_image, 1, 0)
            raw = in_image[0]
            mask = in_image[1]
            # get resolution
            resolution = np.ones(3, dtype=float)
            with tifffile.TiffFile(infn) as tiff:
                image_metadata = tiff.imagej_metadata
                if image_metadata is not None:
                    resolution[0] = image_metadata.get('spacing', 1.)
                else:
                    resolution[0] = 1.
                tags = tiff.pages[0].tags
                num_pixels, units = tags["YResolution"].value
                resolution[1] = units / num_pixels
                num_pixels, units = tags["XResolution"].value
                resolution[2] = units / num_pixels
        else:
            raise NotImplementedError
    
        # check for manual root
        if args.manual_root is not None:
            manual_root = np.array(args.manual_root, dtype=float)
        elif sample_id in manual_roots_left and sample_id in manual_roots_right:
            # heads up: assuming x, y, z here
            l_set_root = np.flip(np.array(manual_roots_left[sample_id]))
            r_set_root = np.flip(np.array(manual_roots_right[sample_id]))
            # resize to pixel space
            l_set_root = np.floor(np.array([l_set_root[0],
                l_set_root[1] / resolution[1],
                l_set_root[2] / resolution[2]])).astype(int)
            r_set_root = np.floor(np.array([r_set_root[0],
                r_set_root[1] / resolution[1],
                r_set_root[2] / resolution[2]])).astype(int)
            manual_root = np.stack([l_set_root, r_set_root], axis=0)
        else:
            manual_root = None
        
        # save raw
        #raw_fn = out_folder + "/%s_raw.tiff" % sample_name
        #io.imsave(raw_fn, raw)
        mip_raw = np.max(raw, axis=0).astype(np.uint8)
        #io.imsave(out_folder + "/%s_raw.png" % sample_name, mip_raw)
        
        # read in mask
        mask = (mask > 0).astype(bool)
        # save mask
        mip = np.max(mask, axis=0).astype(np.uint8) * 255 
        #io.imsave(out_folder + "/%s_mask.png" % sample_name, mip)
        #io.imsave(out_folder + "/%s_mask.tiff" % sample_name, mask)
        #if manual_root is not None:
        #    # set all mask values with lower y coordinate zero
        #    ymin = np.min(manual_root[:, 1])
        #    mask[:, :ymin-10, :] = 0
        if args.mask_cleaned is not None:
            mask_cleaned = io.imread(args.mask_cleaned)
            mask = (mask_cleaned > 0).astype(bool)
            mip = np.max(mask, axis=0).astype(np.uint8) * 255 

        # save raw and mask as tiff
        """
        tmp_zeros = np.zeros_like(raw)
        tmp_mask = (mask * 255).astype(np.uint8)
        tmp_out = np.stack([raw, tmp_mask, tmp_zeros], axis=-1).astype(np.uint8)
        if manual_root is not None:
            for mroot in manual_root:
                tmp_out[mroot[0], mroot[1], mroot[2]] = [255, 0, 255]
        tmp_out = np.moveaxis(tmp_out, -1, 1)
        out_fn = out_folder + ("/%s_raw_mask.tiff" % sample_name)
        # add resolution to tiff file
        # Voxel size: 3.8927x3.8953x8.0132 micron^3
        tifffile.imwrite(out_fn, tmp_out, imagej=True,
                resolution=(1./resolution[2], 1./resolution[1]),
                compression="zlib",
                metadata={
                    "spacing": resolution[0],
                    "unit": "um",
                    "axes": "ZCYX",
                    "mode": "composite"
                    }
                )
        """
        skeletonize_mask(
                mask, 
                out_folder=out_folder,
                sample_name=sample_name,
                dilation_factor=args.dilation_factor,
                max_radius=args.max_radius,
                raw=raw,
                resolution=resolution,
                manual_root=manual_root,
                skel_scale=args.skel_scale,
                only_get_tri=args.only_get_tri,
                tri_thresh=args.tri_thresh
                )


if __name__ == "__main__":
    main()

