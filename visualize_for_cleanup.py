import os
import numpy as np
from skimage import io
import argparse
import json
import h5py
import tifffile
from imaris_ims_file_reader.ims import ims
import napari
from manual_roots import manual_roots_right, manual_roots_left


def get_arguments():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--image", type=str, dest="image", required=True
            )
    parser.add_argument(
            "--dilated_mask", type=str, default=None
            )
    parser.add_argument(
            "--cleaned_mask", type=str, default=None
            )

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    print(args)
    key_raw = "DataSet/ResolutionLevel 2/TimePoint 0/Channel 0/Data"
    key_mask = "DataSet/ResolutionLevel 2/TimePoint 0/Channel 1/Data"

    # read raw and mask image
    if args.image.endswith(".ims"):
        #inf = h5py.File(args.image, "r")
        #raw = np.array(inf[key_raw])
        #mask = (np.array(inf[key_mask]) > 0).astype(np.uint8) * 255
        ds = ims(args.image, ResolutionLevelLock=2) #, resolution_decimal_places=4
        resolution = ds.resolution
        raw = np.array(ds[2,0,0,:,:,:])
        raw = ((raw / float(np.max(raw))) * 255).astype(np.uint8)
        mask = np.array(ds[2,0,1,:,:,:])
        ds.close()
    elif args.image.endswith(".tiff"):
        #in_image = io.imread(args.image)
        #print(in_image.shape)
        #in_image = np.moveaxis(in_image, -1, 0)
        in_image = tifffile.imread(args.image)
        in_image = np.moveaxis(in_image, 1, 0)
        print(in_image.shape)
        raw = in_image[0]
        mask = in_image[1]
        # get resolution
        resolution = np.ones(3, dtype=float)
        with tifffile.TiffFile(args.image) as tiff:
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

    if args.dilated_mask is not None:
        dilated_mask = io.imread(args.dilated_mask)
        print(dilated_mask.shape)
        dilated_mask[dilated_mask > 0] = 1
        dilated_mask = dilated_mask.astype(np.uint8)

    if args.cleaned_mask is not None:
        cleaned_mask = io.imread(args.cleaned_mask)
        print("cleaned mask: ", cleaned_mask.shape)
        cleaned_mask = (cleaned_mask > 0).astype(np.uint8)

    sample_name = os.path.basename(args.image).split(".")[0]
    sample_id = sample_name.split("_")[0]
    if sample_id in manual_roots_left and sample_id in manual_roots_right:
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
        print(l_set_root, r_set_root)
        manual_root = np.stack([l_set_root, r_set_root], axis=0)
        print(manual_root, manual_root.shape)
    else:
        manual_root = None

    mask[mask > 0] = 1
    mask = mask.astype(np.uint8)

    # setup napari viewer
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(raw, colormap="red", name='raw', blending='additive')
    viewer.add_labels(mask, name='mask')
    if manual_root is not None:
        viewer.add_points(
            manual_root,
            ndim=3,
            symbol='o',
            edge_width=0.5,
            size=8,
            border_color="green"
        )
    if args.dilated_mask is not None:
        viewer.add_labels(dilated_mask, name="dilated_mask")
    if args.cleaned_mask is not None:
        viewer.add_labels(cleaned_mask, name="cleaned_mask")

    napari.run()

if __name__ == "__main__":
    main()

