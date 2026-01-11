"""Medical metrics calculation from brain scan segmentation data."""
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import nibabel as nib
import json

from .synthseg_service import TUMOR_LABELS


def calculate_volume_mm3(mask: np.ndarray, spacing: tuple) -> float:
    """Calculate volume in mm³ from a binary mask."""
    voxel_count = np.sum(mask > 0)
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    return float(voxel_count * voxel_volume)


def calculate_midline_shift(data: np.ndarray, spacing: tuple) -> float:
    """
    Calculate midline shift in mm.
    Compares the center of mass of the brain to the geometric center.
    """
    # Create a brain mask (any non-zero voxels)
    brain_mask = data > 0

    if not np.any(brain_mask):
        return 0.0

    # Get the center of mass
    coords = np.where(brain_mask)
    center_of_mass = np.array([np.mean(c) for c in coords])

    # Get the geometric center
    geometric_center = np.array(data.shape) / 2

    # Calculate shift in the x-axis (left-right) in mm
    shift_voxels = abs(center_of_mass[0] - geometric_center[0])
    shift_mm = shift_voxels * spacing[0]

    return float(shift_mm)


def calculate_infiltration_index(
    edema_volume: float,
    tumor_core_volume: float
) -> Optional[float]:
    """
    Calculate infiltration index.
    Ratio of edema volume to tumor core volume.
    Higher values may indicate more aggressive behavior.
    """
    if tumor_core_volume <= 0:
        return None
    return float(edema_volume / tumor_core_volume)


def calculate_tumor_metrics_from_segmentation(
    segmentation_path: Path
) -> Dict[str, Any]:
    """
    Calculate tumor metrics from a segmentation NIfTI file.

    Returns metrics including:
    - Total lesion volume
    - Active enhancing volume
    - Necrotic core volume
    - Edema volume
    - Midline shift
    - Infiltration index
    """
    try:
        img = nib.load(str(segmentation_path))
        data = img.get_fdata().astype(np.int32)
        spacing = img.header.get_zooms()[:3]
    except Exception as e:
        print(f"Error loading segmentation: {e}")
        return {"has_tumor": False, "error": str(e)}

    # Check for tumor labels
    unique_labels = np.unique(data).tolist()

    # BraTS tumor labels (100-103)
    necrotic_label = 100  # Necrotic tumor core
    edema_label = 101     # Peritumoral edema
    enhancing_label = 102  # Enhancing tumor
    non_enhancing_label = 103  # Non-enhancing tumor

    has_tumor = any(label in unique_labels for label in [necrotic_label, edema_label, enhancing_label, non_enhancing_label])

    if not has_tumor:
        # Check for simple tumor labels (1-4 in some formats)
        simple_tumor_labels = [1, 2, 3, 4]
        has_simple_tumor = any(label in unique_labels for label in simple_tumor_labels)
        if has_simple_tumor and len(unique_labels) <= 5:  # Likely a tumor segmentation
            has_tumor = True
            # Map simple labels
            necrotic_label = 1
            edema_label = 2
            enhancing_label = 3
            non_enhancing_label = 4

    if not has_tumor:
        return {"has_tumor": False}

    # Calculate volumes
    necrotic_mask = data == necrotic_label
    edema_mask = data == edema_label
    enhancing_mask = data == enhancing_label
    non_enhancing_mask = data == non_enhancing_label

    necrotic_volume = calculate_volume_mm3(necrotic_mask, spacing)
    edema_volume = calculate_volume_mm3(edema_mask, spacing)
    enhancing_volume = calculate_volume_mm3(enhancing_mask, spacing)
    non_enhancing_volume = calculate_volume_mm3(non_enhancing_mask, spacing)

    # Total lesion = all tumor components
    total_tumor_mask = necrotic_mask | enhancing_mask | non_enhancing_mask
    total_lesion_volume = calculate_volume_mm3(total_tumor_mask, spacing)

    # Tumor core for infiltration index (excluding edema)
    tumor_core_volume = necrotic_volume + enhancing_volume + non_enhancing_volume

    # Calculate midline shift
    midline_shift = calculate_midline_shift(data, spacing)

    # Calculate infiltration index
    infiltration_index = calculate_infiltration_index(edema_volume, tumor_core_volume)

    return {
        "has_tumor": True,
        "total_lesion_volume_mm3": total_lesion_volume,
        "active_enhancing_volume_mm3": enhancing_volume,
        "necrotic_core_volume_mm3": necrotic_volume,
        "edema_volume_mm3": edema_volume,
        "non_enhancing_volume_mm3": non_enhancing_volume,
        "midline_shift_mm": midline_shift,
        "infiltration_index": infiltration_index,
        "tumor_core_volume_mm3": tumor_core_volume,
    }


def get_scan_regions_from_metadata(metadata_path: Path) -> List[str]:
    """Extract region names from scan metadata."""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return [r.get('label', r.get('name', 'Unknown')) for r in metadata.get('regions', [])]
    except Exception:
        return []


def aggregate_case_metrics(scan_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple scans to show progression.
    """
    if not scan_metrics:
        return {"has_tumor": False, "scan_count": 0}

    # Check if any scan has tumor
    has_tumor = any(m.get('has_tumor', False) for m in scan_metrics)

    if not has_tumor:
        return {"has_tumor": False, "scan_count": len(scan_metrics)}

    # Get metrics from scans with tumor data
    tumor_scans = [m for m in scan_metrics if m.get('has_tumor')]

    if not tumor_scans:
        return {"has_tumor": False, "scan_count": len(scan_metrics)}

    # Latest metrics
    latest = tumor_scans[-1]

    # Calculate progression if multiple tumor scans
    progression = None
    if len(tumor_scans) >= 2:
        first = tumor_scans[0]
        latest_volume = latest.get('total_lesion_volume_mm3', 0)
        first_volume = first.get('total_lesion_volume_mm3', 0)

        if first_volume > 0:
            volume_change = latest_volume - first_volume
            volume_change_percent = (volume_change / first_volume) * 100
            progression = {
                "initial_volume_mm3": first_volume,
                "current_volume_mm3": latest_volume,
                "absolute_change_mm3": volume_change,
                "percent_change": volume_change_percent,
                "trend": "increasing" if volume_change > 0 else "decreasing" if volume_change < 0 else "stable"
            }

    return {
        "has_tumor": True,
        "scan_count": len(scan_metrics),
        "tumor_scan_count": len(tumor_scans),
        "latest_metrics": latest,
        "progression": progression,
    }
