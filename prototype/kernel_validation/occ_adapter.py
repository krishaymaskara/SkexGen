"""OpenCascade adapter; this is the only module that imports PythonOCC."""

from __future__ import annotations

import OCC
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop_VolumeProperties
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Core.GProp import GProp_GProps
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Vec
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods

from .adapter import KernelAdapterError, WorldArc, WorldCircle, WorldLine
from .model import InspectedShape, ShapeMetrics

try:
    from OCC.Core.Standard import Standard_Version
except ImportError:  # Binding availability differs across PythonOCC releases.
    Standard_Version = None


class OpenCascadeAdapter:
    """Small, non-healing wrapper around OpenCascade construction calls."""

    def backend_versions(self):
        return {
            "pythonocc_version": getattr(OCC, "VERSION", getattr(OCC, "__version__", None)),
            "opencascade_version": str(Standard_Version()) if Standard_Version else None,
        }

    def make_wire(self, primitives):
        maker = BRepBuilderAPI_MakeWire()
        for primitive in primitives:
            maker.Add(self._edge(primitive))
        maker.Build()
        if not maker.IsDone():
            raise KernelAdapterError("OpenCascade did not construct the sketch wire")
        return maker.Wire()

    def wire_is_closed(self, wire):
        return bool(BRep_Tool.IsClosed(wire))

    def shape_is_valid(self, shape):
        return not shape.IsNull() and bool(BRepCheck_Analyzer(shape).IsValid())

    def make_face(self, wire):
        maker = BRepBuilderAPI_MakeFace(wire, True)
        if not maker.IsDone():
            raise KernelAdapterError("OpenCascade did not construct the profile face")
        return maker.Face()

    def make_axis(self, point, direction):
        try:
            return gp_Ax1(gp_Pnt(*point), gp_Dir(*direction))
        except Exception as exc:
            raise KernelAdapterError("OpenCascade rejected the revolve axis") from exc

    def extrude(self, face, vector):
        maker = BRepPrimAPI_MakePrism(face, gp_Vec(*vector), False, True)
        maker.Build()
        if not maker.IsDone():
            raise KernelAdapterError("OpenCascade extrusion did not complete")
        return maker.Shape()

    def revolve(self, face, axis, angle_radians):
        maker = BRepPrimAPI_MakeRevol(face, axis, angle_radians, True)
        maker.Build()
        if not maker.IsDone():
            raise KernelAdapterError("OpenCascade revolution did not complete")
        return maker.Shape()

    def join(self, current, feature):
        maker = BRepAlgoAPI_Fuse(current, feature)
        maker.Build()
        if not maker.IsDone():
            raise KernelAdapterError("OpenCascade Boolean JOIN did not complete")
        return maker.Shape()

    def cut(self, current, feature):
        maker = BRepAlgoAPI_Cut(current, feature)
        maker.Build()
        if not maker.IsDone():
            raise KernelAdapterError("OpenCascade Boolean CUT did not complete")
        return maker.Shape()

    def inspect_solid(self, shape):
        is_null = shape is None or shape.IsNull()
        if is_null:
            return InspectedShape(shape, ShapeMetrics(True, False, None, None, None, 0, 0, 0, 0))
        solids = self._subshapes(shape, TopAbs_SOLID)
        inspected_shape = solids[0] if len(solids) == 1 else shape
        valid = bool(BRepCheck_Analyzer(shape).IsValid())
        volume = None
        bounding_box = None
        if len(solids) == 1:
            properties = GProp_GProps()
            brepgprop_VolumeProperties(inspected_shape, properties)
            volume = float(properties.Mass())
            box = Bnd_Box()
            brepbndlib_Add(inspected_shape, box)
            if not box.IsVoid():
                bounding_box = tuple(float(item) for item in box.Get())
        metrics = ShapeMetrics(
            False,
            valid,
            "solid" if len(solids) == 1 else _shape_type_name(shape.ShapeType()),
            volume,
            bounding_box,
            len(solids),
            len(self._subshapes(inspected_shape, TopAbs_FACE)),
            len(self._subshapes(inspected_shape, TopAbs_EDGE)),
            len(self._subshapes(inspected_shape, TopAbs_VERTEX)),
        )
        return InspectedShape(inspected_shape, metrics)

    def _edge(self, primitive):
        if isinstance(primitive, WorldLine):
            maker = BRepBuilderAPI_MakeEdge(gp_Pnt(*primitive.start), gp_Pnt(*primitive.end))
        elif isinstance(primitive, WorldArc):
            curve = GC_MakeArcOfCircle(
                gp_Pnt(*primitive.start), gp_Pnt(*primitive.midpoint), gp_Pnt(*primitive.end)
            )
            if not curve.IsDone():
                raise KernelAdapterError("OpenCascade rejected a sketch arc")
            maker = BRepBuilderAPI_MakeEdge(curve.Value())
        elif isinstance(primitive, WorldCircle):
            circle = gp_Circ(
                gp_Ax2(gp_Pnt(*primitive.center), gp_Dir(*primitive.normal), gp_Dir(*primitive.x_direction)),
                primitive.radius,
            )
            maker = BRepBuilderAPI_MakeEdge(circle)
        else:
            raise KernelAdapterError(f"unsupported world primitive {type(primitive).__name__}")
        if not maker.IsDone():
            raise KernelAdapterError("OpenCascade did not construct a sketch edge")
        return maker.Edge()

    @staticmethod
    def _subshapes(shape, kind):
        explorer = TopExp_Explorer(shape, kind)
        result = []
        while explorer.More():
            current = explorer.Current()
            result.append(topods.Solid(current) if kind == TopAbs_SOLID else current)
            explorer.Next()
        return result


def _shape_type_name(shape_type):
    return {
        TopAbs_SOLID: "solid",
        TopAbs_FACE: "face",
        TopAbs_EDGE: "edge",
        TopAbs_VERTEX: "vertex",
    }.get(shape_type, "other")
