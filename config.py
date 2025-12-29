
PLC_DEFECT_BITS = {
    "circularity": 0,
    "flash": 1,
    "short_fill": 2,
    "pinholes": 3,
    "blobs": 4,
    "color": 5,
    "chips": 6,
    "cracks": 7,
}

CFG = {
    'resize_width': 1024,
    "min_circularity": 0.92,
    "max_flash_ratio": 0.015,
    "min_fill_ratio": 0.97,
    "max_hull_deficit": 0.01,
    "min_blob_area": 5,
    "max_blob_area": 200,
    "min_crack_length": 25,
    "max_chip_depth_ratio": 0.02,
    "color_delta_thresh": 18,
}

COLOR_CFG = {
    # reference storage
    "reference_save_path": "color_references.json",

    # center / edge logic
    "center_erosion_ratio": 0.35,

    # patch logic
    "patch_grid_size": (32, 32),      # (h, w)
    "patch_delta_e_threshold": 6.0,
    "patch_min_area_px": 80,

    # decision thresholds
    "global_delta_e_threshold": 5.0,
    "edge_delta_e_threshold": 5.0,

    # debug drawing
    "debug_draw_font_scale": 0.5,
    "debug_draw_thickness": 1,
}