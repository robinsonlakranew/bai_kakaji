
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
    'gaussian_blur_ksize': (5, 5),
    'min_contour_area': 2000,        # reject tiny contours
    'circularity_tol': 0.85,        # threshold for 4πA/P²
    'axis_ratio_tol': 0.95,         # minor/major (ellipse)
    'radial_variance_tol': 0.035,   # normalized variance
    'require_checks': 2,            # how many of the 3 checks must pass
    # small blob (optional) settings:
    'pinhole_thresh_blocksize': 41,
    'pinhole_thresh_C': 8,
    'small_blob_area_min': 20,
    'small_blob_area_max': 800,
    'pinhole_area_max': 200,
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

DEFAULT_CFG = {
    "gaussian_blur_ksize": (5, 5),
    "patch_grid_size": (20, 20),        # patch (h,w) in pixels for grid — used if patch_size is int -> square
    "patch_min_area_px": 12,            # minimum blob area in pixels to keep
    "global_delta_e_threshold": 12.0,   # threshold for global ΔE (CIE76) to mark whole-cap drift
    "patch_delta_e_threshold": 18.0,    # threshold for per-patch ΔE to mark patch contaminated
    "edge_delta_e_threshold": 16.0,     # threshold for edge-specific check
    "center_erosion_ratio": 0.35,       # fraction of mask radius to erode to obtain center region
    "reference_save_path": "cap_color_refs.json",
    "debug_draw_font_scale": 0.6,
    "debug_draw_thickness": 1,
}