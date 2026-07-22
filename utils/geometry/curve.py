# Base data object shared by parsed line, arc, and circle sketch primitives.
import math


class Curve():
    def __init__(self, point_indices, point_data):
        self.point_indices = point_indices
        self.point_geom = point_data[point_indices, 0:2]

    def verts_to_bbox(self, verts):
        # Reduce 2D vertices to axis-aligned minimum and maximum bounds.
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        bbox = [min(xs), max(xs), min(ys), max(ys)]
        return bbox