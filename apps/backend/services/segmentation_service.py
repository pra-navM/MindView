"""Segmentation service utilities."""
from pathlib import Path
from typing import Callable, Optional

from services.mesh_generation import intensity_to_meshes


def fallback_to_intensity(
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    job_id: str,
    progress_callback: Optional[Callable[[int], None]] = None
) -> None:
    """
    Generate intensity-based mesh from raw MRI data.

    This uses the existing intensity_to_meshes function.
    """
    print(f"[{job_id}] Generating intensity-based mesh...")

    intensity_to_meshes(
        input_path,
        output_path,
        metadata_path,
        job_id,
        progress_callback=progress_callback
    )


def get_segmentation_method_name(method: str) -> str:
    """Get human-readable name for segmentation method."""
    methods = {
        "monai_tumor_detection": "MONAI Tumor Detection",
        "brats": "BraTS Tumor Segmentation",
        "simple": "Simple Label Segmentation",
        "intensity": "Intensity-based Layers",
        "pre_segmented": "Pre-segmented Data",
    }
    return methods.get(method, method)
