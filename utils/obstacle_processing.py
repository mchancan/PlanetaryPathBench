"""Obstacle processing helpers: polygonization and clustering for occupancy maps.

Provides:
- polygonize_occ_map(occ_map, resolution=1.0, origin=(0,0), occupied_value=1)
  -> List[List[(x,y)]] polygons (each polygon is list of (x,y) vertices, closed)
- cluster_components_for_sampling(occ_map, resolution=1.0, origin=(0,0),
  occupied_value=1, min_area_pixels=1, merge_distance=2.0)
  -> List[(x, y, radius)] approximating obstacles for sampling planners

Implementation notes:
- Prefer scikit-image's find_contours for smooth contours when available.
- Fallback to scipy.ndimage.label to find connected components and return
  bounding-box polygons when skimage is not present.
- Final fallback: simple per-block downsampling to create coarse circular
  obstacles.
"""
from typing import List, Tuple
import numpy as np


def polygonize_occ_map(occ_map: np.ndarray, resolution: float = 1.0,
                       origin: Tuple[float, float] = (0.0, 0.0),
                       occupied_value=1) -> List[List[Tuple[float, float]]]:
    """Return a list of polygons (list of (x,y) tuples) describing occupied regions.

    Each polygon is closed (first vertex == last vertex).
    Try scikit-image `find_contours` first; otherwise extract connected
    components and return bounding-box polygons.
    Coordinates are in world units (scaled by `resolution` and translated by `origin`).
    """
    occ = np.asarray(occ_map)
    if occ.ndim != 2:
        raise ValueError('occ_map must be 2D')

    # Try scikit-image.find_contours for a good polygon trace
    try:
        from skimage import measure  # type: ignore

        # find_contours traces the 0.5 level set on the boolean image
        binary = (occ == occupied_value).astype(np.uint8)
        # pad array to avoid contours touching edges ambiguously
        padded = np.pad(binary, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded, 0.5)

        polygons: List[List[Tuple[float, float]]] = []
        for contour in contours:
            # contour coordinates are (row, col) in padded image
            # subtract padding and convert to world coords using cell centers
            coords = []
            for r, c in contour:
                r -= 1.0
                c -= 1.0
                x = origin[0] + (c + 0.5) * resolution
                y = origin[1] + (r + 0.5) * resolution
                coords.append((float(x), float(y)))
            if len(coords) >= 3:
                # close polygon
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                polygons.append(coords)

        return polygons
    except Exception:
        # skimage not available or failed -> fallback
        pass

    # Fallback: use scipy.ndimage.label to find connected components and
    # return their axis-aligned bounding boxes as polygons
    try:
        from scipy import ndimage  # type: ignore

        binary = (occ == occupied_value).astype(np.uint8)
        labeled, ncomp = ndimage.label(binary)
        polygons = []
        for lab in range(1, ncomp + 1):
            ys, xs = np.where(labeled == lab)
            if ys.size == 0:
                continue
            min_r, max_r = int(ys.min()), int(ys.max())
            min_c, max_c = int(xs.min()), int(xs.max())
            # bounding box polygon (clockwise) in world coords
            x0 = origin[0] + (min_c + 0.5) * resolution
            y0 = origin[1] + (min_r + 0.5) * resolution
            x1 = origin[0] + (max_c + 0.5) * resolution
            y1 = origin[1] + (max_r + 0.5) * resolution
            poly = [(float(x0), float(y0)), (float(x1), float(y0)),
                    (float(x1), float(y1)), (float(x0), float(y1)),
                    (float(x0), float(y0))]
            polygons.append(poly)
        return polygons
    except Exception:
        pass

    # Final fallback: coarse block sampling -> every kxk block that has
    # any occupied cell becomes a square polygon
    H, W = occ.shape
    bs = max(4, min(H, W) // 128)  # block size depends on map size
    polygons = []
    for r0 in range(0, H, bs):
        for c0 in range(0, W, bs):
            block = occ[r0:min(r0 + bs, H), c0:min(c0 + bs, W)]
            if (block == occupied_value).any():
                r1 = min(r0 + bs - 1, H - 1)
                c1 = min(c0 + bs - 1, W - 1)
                x0 = origin[0] + (c0 + 0.5) * resolution
                y0 = origin[1] + (r0 + 0.5) * resolution
                x1 = origin[0] + (c1 + 0.5) * resolution
                y1 = origin[1] + (r1 + 0.5) * resolution
                poly = [(float(x0), float(y0)), (float(x1), float(y0)),
                        (float(x1), float(y1)), (float(x0), float(y1)),
                        (float(x0), float(y0))]
                polygons.append(poly)
    return polygons


def cluster_components_for_sampling(occ_map: np.ndarray, resolution: float = 1.0,
                                   origin: Tuple[float, float] = (0.0, 0.0),
                                   occupied_value=1,
                                   min_area_pixels: int = 1,
                                   merge_distance: float = 2.0) -> List[Tuple[float, float, float]]:
    """Return a list of circular obstacles (x, y, radius) suitable for sampling planners.

    Strategy:
    - Use connected component labeling (scipy.ndimage if present) to find blobs.
    - For each blob compute bounding box and centroid; approximate radius as half
      of the largest bbox side * resolution.
    - Merge blobs whose centroids are closer than merge_distance (world units).
    - Drop blobs smaller than min_area_pixels.
    """
    occ = np.asarray(occ_map)
    if occ.ndim != 2:
        raise ValueError('occ_map must be 2D')

    H, W = occ.shape

    # Try scipy.ndimage for labeling
    try:
        from scipy import ndimage  # type: ignore

        binary = (occ == occupied_value).astype(np.uint8)
        labeled, ncomp = ndimage.label(binary)
        comps = []
        for lab in range(1, ncomp + 1):
            ys, xs = np.where(labeled == lab)
            area = ys.size
            if area < min_area_pixels:
                continue
            min_r, max_r = int(ys.min()), int(ys.max())
            min_c, max_c = int(xs.min()), int(xs.max())
            cx = origin[0] + (float(xs.mean()) + 0.5) * resolution
            cy = origin[1] + (float(ys.mean()) + 0.5) * resolution
            width = (max_c - min_c + 1) * resolution
            height = (max_r - min_r + 1) * resolution
            radius = 0.5 * max(width, height)
            comps.append({'cx': cx, 'cy': cy, 'r': radius})

        # Merge close components greedily
        merged = []
        while comps:
            base = comps.pop(0)
            bx, by = base['cx'], base['cy']
            br = base['r']
            i = 0
            while i < len(comps):
                other = comps[i]
                dist = ((bx - other['cx']) ** 2 + (by - other['cy']) ** 2) ** 0.5
                if dist <= merge_distance + br + other['r']:
                    # merge: compute bounding box of both
                    minx = min(bx - br, other['cx'] - other['r'])
                    miny = min(by - br, other['cy'] - other['r'])
                    maxx = max(bx + br, other['cx'] + other['r'])
                    maxy = max(by + br, other['cy'] + other['r'])
                    bx = (minx + maxx) / 2.0
                    by = (miny + maxy) / 2.0
                    br = max((maxx - minx), (maxy - miny)) / 2.0
                    comps.pop(i)
                    i = 0
                else:
                    i += 1
            merged.append((bx, by, br))

        return merged
    except Exception:
        # Fallback: downsample grid into blocks
        bs = max(4, min(H, W) // 256)
        circles = []
        for r0 in range(0, H, bs):
            for c0 in range(0, W, bs):
                block = occ[r0:min(r0 + bs, H), c0:min(c0 + bs, W)]
                if (block == occupied_value).any():
                    cx = origin[0] + (c0 + min(bs, W - c0) / 2.0) * resolution
                    cy = origin[1] + (r0 + min(bs, H - r0) / 2.0) * resolution
                    radius = 0.5 * max(min(bs, W - c0) * resolution, min(bs, H - r0) * resolution)
                    circles.append((float(cx), float(cy), float(radius)))
        return circles
