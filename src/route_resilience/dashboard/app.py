"""Route Resilience dashboard (M10) — Streamlit + Leaflet decision support.

Thin UI over dashboard/service.py (all logic is tested there). Matches roadmap §15:
a real OSM basemap with the extracted+healed road graph, betweenness criticality
heat, and interactive hazard simulation — click a junction to "flood" it and read
the live Resilience Index, the recomputed betweenness, and rerouting cost.

Run:
    conda activate route-resilience
    streamlit run src/route_resilience/dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from route_resilience.config import load_config
from route_resilience.dashboard import service
from route_resilience.paths import PROCESSED
from route_resilience.resilience.centrality import node_betweenness

st.set_page_config(page_title="Route Resilience", layout="wide")

cfg = load_config("base.yaml", "data.yaml", "graph.yaml")
RES = float(cfg.data.resolution_m)


@st.cache_resource(show_spinner="Building graph…")
def _load_graph(mask_path: str, heal: bool, max_gap: float):
    g, crs = service.load_tile_graph(mask_path, RES, heal=heal, max_gap_m=max_gap)
    return g, crs, node_betweenness(g)


@st.cache_data
def _manifest() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "manifest.csv").sort_values("road_frac", ascending=False)


# ----------------------------- sidebar controls -----------------------------
st.sidebar.title("🛰️ Route Resilience")
st.sidebar.caption("Occlusion-robust extraction → routable graph → resilience twin")

man = _manifest()
labels = [f"{r.place} · {r.terrain} · {r.road_frac:.1%} road" for r in man.itertuples()]
choice = st.sidebar.selectbox("Tile", range(len(labels)), format_func=lambda i: labels[i])
row = man.iloc[choice]

heal = st.sidebar.checkbox("Heal graph gaps", value=True)
max_gap = st.sidebar.slider("Healing max gap (m)", 10.0, 100.0,
                            float(cfg.graph.healing.max_gap_m), 5.0)
radius = st.sidebar.slider("Flood radius (m)", 10.0, 200.0, 50.0, 5.0)
click_mode = st.sidebar.radio("A map click sets…", ["Flood center", "Route start", "Route end"])
if st.sidebar.button("Reset selection"):
    for k in ("flood", "src", "dst", "last_click"):
        st.session_state.pop(k, None)

g, crs, bc_base = _load_graph(row["mask_path"], heal, max_gap)

if g.number_of_nodes() == 0:
    st.warning("This tile produced an empty graph. Pick another tile.")
    st.stop()

# ----------------------------- handle map click -----------------------------
def _assign_click(lat: float, lon: float) -> None:
    node = service.nearest_node(g, lon, lat, crs)
    if click_mode == "Flood center":
        st.session_state["flood"] = node
    elif click_mode == "Route start":
        st.session_state["src"] = node
    else:
        st.session_state["dst"] = node


flood_node = st.session_state.get("flood")
src, dst = st.session_state.get("src"), st.session_state.get("dst")

# ----------------------------- run the twin ---------------------------------
impacted, bc_show, report = set(), bc_base, None
if flood_node is not None and flood_node in g:
    report, impacted, bc_show = service.flood(g, flood_node, radius)

route = None
if src in g and dst in g and src is not None and dst is not None:
    route = service.reroute(g, src, dst, ablated_nodes=impacted)

hazard_center = None
if flood_node is not None and flood_node in g:
    hazard_center = (g.nodes[flood_node]["lat"], g.nodes[flood_node]["lon"])

m = service.build_map(
    g, bc_show, impacted=impacted,
    before_path=route["before_path"] if route else None,
    after_path=route["after_path"] if route else None,
    hazard_center=hazard_center, hazard_radius_m=radius if hazard_center else None,
)

# ----------------------------- layout ---------------------------------------
st.title("Urban Road Resilience — Decision Support")
left, right = st.columns([3, 1])

with left:
    map_data = st_folium(m, width=None, height=620, returned_objects=["last_clicked"])
    lc = (map_data or {}).get("last_clicked")
    if lc and st.session_state.get("last_click") != (lc["lat"], lc["lng"]):
        st.session_state["last_click"] = (lc["lat"], lc["lng"])
        _assign_click(lc["lat"], lc["lng"])
        st.rerun()

with right:
    st.subheader("Network")
    st.metric("Nodes / Edges", f"{g.number_of_nodes()} / {g.number_of_edges()}")
    st.caption(f"CRS {crs} · {row.terrain}")

    if report is not None:
        st.subheader("Resilience under flood")
        st.metric("Resilience Index", f"{report['resilience_index']:.2f}",
                  delta=f"-{report['efficiency_drop'] * 100:.0f}% efficiency",
                  delta_color="inverse")
        st.metric("Junctions flooded", report["n_impacted"])
        st.metric("Components after", report["components_after"])
    else:
        st.info("Click a junction (mode: **Flood center**) to simulate a closure.")

    if route is not None:
        st.subheader("Reroute")
        c = route["cost"]
        before = "∞" if c["before"] == float("inf") else f"{c['before']:.0f} m"
        after = "∞ (cut off)" if c["after"] == float("inf") else f"{c['after']:.0f} m"
        st.metric("Path before → after", f"{before} → {after}")
        if c["ratio"] not in (None, float("inf")):
            st.caption(f"Detour factor ×{c['ratio']:.2f}")

st.caption("Set two points (Route start/end) and a flood center to see live rerouting. "
           "Red dashed = healed bridges · red ✕ = flooded junctions.")
