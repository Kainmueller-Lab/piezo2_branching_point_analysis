# piezo2_branching_point_analysis
Scripts for skeletonization and branching point analysis

## run branching point analysis
```bash
python main.py --in_file <path-to-image-file> --mask_cleaned <path-to-segmentation-mask> --dilation_factor 7 --skel_scale 7
```

## run napari script to manually clean up segmentation mask
```bash
python visualize_for_cleanup.py --image <path-to-image-file>
```
