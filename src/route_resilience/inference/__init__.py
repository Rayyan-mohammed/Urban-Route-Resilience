"""Phase I->II bridge — run a trained model over real imagery to get a mask.

This is the finale integration surface: Cartosat-3 imagery arrives as GeoTIFFs,
`predict_geotiff` turns them into geo-referenced road masks, and the existing
graph -> heal -> twin -> dashboard chain consumes them unchanged.
"""

from .predict import (
    load_model_from_checkpoint,
    predict_array,
    predict_geotiff,
    to_uint8_rgb,
)

__all__ = [
    "load_model_from_checkpoint",
    "predict_array",
    "predict_geotiff",
    "to_uint8_rgb",
]
