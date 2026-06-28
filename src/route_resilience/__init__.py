"""Route Resilience — occlusion-robust road extraction + resilience digital twin.

Package layout (mirrors the roadmap's 4-phase architecture):

    data/        Phase 0  — tiling, OSM auto-masks, synthetic occlusion, Dataset
    models/      Phase I  — baseline (U-Net) + SegFormer-B2, clDice & losses
    training/    Phase I  — train/val loop, schedulers, checkpointing
    evaluation/  §11      — clDice, occlusion-recall, APLS, connectivity ratio
    graph/       Phase II — skeletonize -> NetworkX graph -> healing
    resilience/  Phase III— centrality, hazard layer, dynamic ablation, R index
    dashboard/   Phase IV — Streamlit + Leaflet decision-support UI
    utils/                — logging, IO, geometry helpers
"""

__version__ = "0.1.0"
