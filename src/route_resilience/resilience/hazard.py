"""Hazard layer (M8) — what gets 'flooded' to stress-test the network.

A Hazard maps a graph to the set of impacted nodes. Synthetic generators are used
now; a real DEM/flood raster implements the SAME `impacted_nodes` interface later
(returning low-lying nodes) with zero downstream change — that's the pluggability
the roadmap's hazard-grounded story needs.

Nodes carry world (x, y) from the tile transform, so spatial hazards work in metres.
"""

from __future__ import annotations

import networkx as nx


class Hazard:
    """Interface: return the set of node ids a hazard impacts."""

    def impacted_nodes(self, g: nx.Graph) -> set:
        raise NotImplementedError


class NodeHazard(Hazard):
    """Explicit node set (e.g. a junction the planner clicked)."""

    def __init__(self, nodes):
        self.nodes = set(nodes)

    def impacted_nodes(self, g: nx.Graph) -> set:
        return {n for n in self.nodes if n in g}


class RadiusHazard(Hazard):
    """All nodes within `radius_m` of a world point (a localized flood)."""

    def __init__(self, center: tuple[float, float], radius_m: float):
        self.cx, self.cy = center
        self.r2 = radius_m * radius_m

    def impacted_nodes(self, g: nx.Graph) -> set:
        out = set()
        for n, d in g.nodes(data=True):
            if (d["x"] - self.cx) ** 2 + (d["y"] - self.cy) ** 2 <= self.r2:
                out.add(n)
        return out


class BandHazard(Hazard):
    """All nodes whose `axis` ('x' or 'y') coordinate is in [lo, hi].

    Stands in for a low-lying strip / river corridor until a DEM is wired up.
    """

    def __init__(self, axis: str = "y", lo: float = 0.0, hi: float = 0.0):
        self.axis = axis
        self.lo = lo
        self.hi = hi

    def impacted_nodes(self, g: nx.Graph) -> set:
        return {n for n, d in g.nodes(data=True) if self.lo <= d[self.axis] <= self.hi}


class RasterHazard(Hazard):
    """Real hazard grounded in a raster — a DEM or a flood-depth layer.

    This is the production form of the interface the synthetic hazards above
    stand in for: it samples the raster at each junction's world coordinate and
    decides whether that junction is out of service.

        # Flood-depth product: junctions under more than 0.3 m of water.
        RasterHazard("flood_depth.tif", mode="depth", threshold=0.3)

        # Bare DEM: junctions below the 812 m flood water level.
        RasterHazard("dem.tif", mode="elevation", threshold=812.0)

    Node coordinates are reprojected to the raster CRS when `node_crs` is given,
    so the DEM does not have to share the imagery's projection. Nodes outside the
    raster, or on nodata, are treated as unaffected — a hazard layer that does not
    cover a junction is not evidence that the junction is flooded.

    rasterio/pyproj are imported lazily so the resilience package keeps working
    (synthetic hazards and all) on machines without the geospatial stack.
    """

    def __init__(
        self,
        raster_path,
        *,
        mode: str = "depth",
        threshold: float = 0.0,
        band: int = 1,
        node_crs: str | None = None,
    ):
        import rasterio

        if mode not in ("depth", "elevation"):
            raise ValueError(f"mode must be 'depth' or 'elevation', got {mode!r}")
        self.mode = mode
        self.threshold = float(threshold)
        self.raster_path = str(raster_path)

        with rasterio.open(raster_path) as ds:
            self.values = ds.read(band).astype("float64")
            self.inv_transform = ~ds.transform
            self.height, self.width = ds.height, ds.width
            self.nodata = ds.nodata
            self.raster_crs = str(ds.crs) if ds.crs else None

        self._to_raster = None
        if node_crs and self.raster_crs and str(node_crs) != self.raster_crs:
            from pyproj import Transformer

            self._to_raster = Transformer.from_crs(
                str(node_crs), self.raster_crs, always_xy=True
            ).transform

    def sample(self, x: float, y: float) -> float | None:
        """Raster value at a world coordinate, or None if outside / nodata."""
        if self._to_raster is not None:
            x, y = self._to_raster(x, y)
        col, row = self.inv_transform * (x, y)
        r, c = int(row), int(col)
        if not (0 <= r < self.height and 0 <= c < self.width):
            return None
        v = float(self.values[r, c])
        if self.nodata is not None and v == float(self.nodata):
            return None
        if v != v:                                   # NaN nodata
            return None
        return v

    def _is_hit(self, v: float) -> bool:
        # Depth: standing water deeper than the threshold closes the junction.
        # Elevation: ground below the flood water level is inundated.
        return v > self.threshold if self.mode == "depth" else v <= self.threshold

    def impacted_nodes(self, g: nx.Graph) -> set:
        out = set()
        for n, d in g.nodes(data=True):
            v = self.sample(d["x"], d["y"])
            if v is not None and self._is_hit(v):
                out.add(n)
        return out
