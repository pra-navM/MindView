"""Atlas-based brain segmentation using ANTsPy registration."""
import json
from pathlib import Path
from typing import Optional

import ants
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
import trimesh


# Atlas directory
BASE_DIR = Path(__file__).parent.parent
ATLAS_DIR = BASE_DIR / "storage" / "atlases"

# Atlas file paths
MNI_TEMPLATE_PATH = ATLAS_DIR / "MNI152_T1_1mm_brain.nii.gz"
CORTICAL_ATLAS_PATH = ATLAS_DIR / "HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"
SUBCORTICAL_ATLAS_PATH = ATLAS_DIR / "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"


# Region definitions with colors (RGBA)
REGION_DEFINITIONS = {
    "brain_shell": {
        "label": "Brain Shell",
        "color": [180, 180, 180, 80],
        "opacity": 0.3,
        "source": "intensity",  # Generated from intensity threshold
        "defaultVisible": True,
    },
    "ventricles": {
        "label": "Ventricles",
        "color": [100, 150, 255, 200],
        "opacity": 0.8,
        "source": "subcortical",
        "atlas_labels": [3, 14],  # Lateral ventricles
        "defaultVisible": True,
    },
    "frontal_lobe": {
        "label": "Frontal Lobe",
        "color": [255, 180, 180, 180],
        "opacity": 0.7,
        "source": "cortical",
        "atlas_labels": [1, 2, 3, 4, 5, 6],
        "defaultVisible": True,
    },
    "temporal_lobe": {
        "label": "Temporal Lobe",
        "color": [180, 255, 180, 180],
        "opacity": 0.7,
        "source": "cortical",
        "atlas_labels": [9, 10, 15, 16],
        "defaultVisible": True,
    },
    "parietal_lobe": {
        "label": "Parietal Lobe",
        "color": [180, 180, 255, 180],
        "opacity": 0.7,
        "source": "cortical",
        "atlas_labels": [7, 8, 17, 18],
        "defaultVisible": True,
    },
    "occipital_lobe": {
        "label": "Occipital Lobe",
        "color": [255, 255, 180, 180],
        "opacity": 0.7,
        "source": "cortical",
        "atlas_labels": [19, 20, 21, 22],
        "defaultVisible": True,
    },
    "tumor": {
        "label": "Tumor",
        "color": [255, 80, 80, 255],
        "opacity": 1.0,
        "source": "segmentation",  # From input segmentation if present
        "defaultVisible": True,
    },
}


def check_atlas_files() -> bool:
    """Check if all required atlas files are present."""
    print(f"Checking atlas files in: {ATLAS_DIR}")
    print(f"ATLAS_DIR exists: {ATLAS_DIR.exists()}")
    required_files = [MNI_TEMPLATE_PATH, CORTICAL_ATLAS_PATH, SUBCORTICAL_ATLAS_PATH]
    for f in required_files:
        print(f"  {f.name}: exists={f.exists()}")
    missing = [f for f in required_files if not f.exists()]
    if missing:
        print(f"Missing atlas files: {missing}")
        return False
    print("All atlas files found!")
    return True


def register_atlas_to_patient(patient_mri_path: str, job_id: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Register atlas labels to patient MRI space using ANTsPy affine registration.

    Returns:
        Tuple of (cortical_labels, subcortical_labels) in patient space, or (None, None) if failed
    """
    if not check_atlas_files():
        print(f"[{job_id}] Atlas files not found, skipping atlas registration")
        return None, None

    try:
        print(f"[{job_id}] Loading patient MRI...")
        patient_mri = ants.image_read(str(patient_mri_path))

        print(f"[{job_id}] Loading MNI template...")
        mni_template = ants.image_read(str(MNI_TEMPLATE_PATH))

        print(f"[{job_id}] Loading atlas labels...")
        cortical_atlas = ants.image_read(str(CORTICAL_ATLAS_PATH))
        subcortical_atlas = ants.image_read(str(SUBCORTICAL_ATLAS_PATH))

        print(f"[{job_id}] Performing affine registration (this may take 30-60 seconds)...")
        registration = ants.registration(
            fixed=patient_mri,
            moving=mni_template,
            type_of_transform="Affine"
        )

        print(f"[{job_id}] Applying transforms to cortical atlas...")
        cortical_in_patient = ants.apply_transforms(
            fixed=patient_mri,
            moving=cortical_atlas,
            transformlist=registration["fwdtransforms"],
            interpolator="nearestNeighbor"  # Preserve integer labels
        )

        print(f"[{job_id}] Applying transforms to subcortical atlas...")
        subcortical_in_patient = ants.apply_transforms(
            fixed=patient_mri,
            moving=subcortical_atlas,
            transformlist=registration["fwdtransforms"],
            interpolator="nearestNeighbor"
        )

        print(f"[{job_id}] Atlas registration complete")
        return cortical_in_patient.numpy(), subcortical_in_patient.numpy()

    except Exception as e:
        print(f"[{job_id}] Atlas registration failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_region_mask(atlas_data: np.ndarray, label_ids: list[int]) -> np.ndarray:
    """Create a binary mask from atlas data for specific label IDs."""
    return np.isin(atlas_data, label_ids).astype(np.float32)


def mask_to_mesh(
    mask: np.ndarray,
    spacing: tuple,
    color: list[int],
    job_id: str,
    region_name: str,
    max_faces: int = 100000,
    smoothing_iterations: int = 10
) -> Optional[trimesh.Trimesh]:
    """Convert a binary mask to a smoothed mesh with color."""
    try:
        # Apply light Gaussian smoothing to reduce voxel artifacts
        mask_smoothed = gaussian_filter(mask, sigma=0.5)

        verts, faces, normals, _ = marching_cubes(
            mask_smoothed,
            level=0.5,
            spacing=spacing
        )

        if len(verts) == 0:
            print(f"[{job_id}] {region_name}: No vertices generated")
            return None

        print(f"[{job_id}] {region_name}: {len(verts)} vertices, {len(faces)} faces")

        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals
        )

        # Apply Laplacian smoothing
        trimesh.smoothing.filter_laplacian(mesh, iterations=smoothing_iterations)

        # Simplify if needed
        if len(mesh.faces) > max_faces:
            try:
                mesh = mesh.simplify_quadric_decimation(max_faces)
                print(f"[{job_id}] {region_name}: Simplified to {len(mesh.faces)} faces")
            except Exception as e:
                print(f"[{job_id}] {region_name}: Simplification failed: {e}")

        # Apply vertex colors
        vertex_colors = np.tile(color, (len(mesh.vertices), 1)).astype(np.uint8)
        mesh.visual.vertex_colors = vertex_colors

        # Store region name in mesh metadata
        mesh.metadata["region_name"] = region_name

        return mesh

    except Exception as e:
        print(f"[{job_id}] {region_name} mesh generation failed: {e}")
        return None


def generate_brain_shell_mesh(
    intensity_data: np.ndarray,
    spacing: tuple,
    job_id: str
) -> Optional[trimesh.Trimesh]:
    """Generate the outer brain shell from intensity data."""
    smoothed = gaussian_filter(intensity_data, sigma=1.0)
    non_zero = smoothed[smoothed > 0]

    if len(non_zero) == 0:
        return None

    # Use 25th percentile for outer surface
    threshold = np.percentile(non_zero, 25)
    mask = (smoothed > threshold).astype(np.float32)

    color = REGION_DEFINITIONS["brain_shell"]["color"]
    return mask_to_mesh(mask, spacing, color, job_id, "brain_shell", max_faces=150000)


def detect_tumor_from_segmentation(seg_data: np.ndarray, all_labels: list[int]) -> Optional[np.ndarray]:
    """
    Detect tumor regions from segmentation data.
    Only flags as tumor if the segmentation appears to be BraTS format (has labels 1, 2, 4 but NOT 3).
    """
    # BraTS convention: labels 1, 2, 4 are tumor regions (label 3 is not used in BraTS)
    # If label 3 exists, this is likely NOT a BraTS tumor segmentation
    brats_tumor_labels = [1, 2, 4]

    # Check if this looks like BraTS format
    has_brats_labels = any(l in all_labels for l in brats_tumor_labels)
    has_label_3 = 3 in all_labels

    # Only treat as tumor if it looks like BraTS format (has 1,2,4 but not 3)
    # and doesn't have too many unique labels (BraTS has max 4 labels: 0,1,2,4)
    is_likely_brats = has_brats_labels and not has_label_3 and len(all_labels) <= 4

    if is_likely_brats:
        tumor_mask = np.isin(seg_data, brats_tumor_labels)
        if np.sum(tumor_mask) > 0:
            return tumor_mask.astype(np.float32)

    return None


def _generate_intensity_layers(
    job_id: str,
    data: np.ndarray,
    spacing: tuple,
    progress_callback=None
) -> list[trimesh.Trimesh]:
    """Generate meshes from intensity data using multiple thresholds (fallback when atlas not available)."""
    smoothed = gaussian_filter(data, sigma=1.0)
    print(f"[{job_id}] Smoothing complete")

    non_zero = smoothed[smoothed > 0]
    if len(non_zero) == 0:
        return []

    # Use thresholds that better separate tissue types
    thresholds = [
        (np.percentile(non_zero, 25), [150, 180, 210, 200]),  # Outer surface
        (np.percentile(non_zero, 50), [100, 150, 200, 220]),  # Mid layer
        (np.percentile(non_zero, 70), [80, 120, 180, 240]),   # Inner structure
        (np.percentile(non_zero, 85), [200, 100, 100, 255]),  # Deep structures
    ]
    print(f"[{job_id}] Thresholds: {[t[0] for t in thresholds]}")

    meshes = []

    for i, (threshold_value, color) in enumerate(thresholds):
        if progress_callback:
            progress_callback(30 + (i * 15))
        print(f"[{job_id}] Processing layer {i+1}/4 at threshold {threshold_value:.2f}")

        try:
            verts, faces, normals, _ = marching_cubes(
                smoothed,
                level=threshold_value,
                spacing=spacing
            )
            print(f"[{job_id}] Layer {i+1}: {len(verts)} vertices, {len(faces)} faces")

            if len(verts) == 0:
                print(f"[{job_id}] Layer {i+1}: No vertices, skipping")
                continue

            mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                vertex_normals=normals
            )

            # Apply Laplacian smoothing
            trimesh.smoothing.filter_laplacian(mesh, iterations=10)

            # Simplify if too many faces
            if len(mesh.faces) > 300000:
                try:
                    mesh = mesh.simplify_quadric_decimation(300000)
                    print(f"[{job_id}] Layer {i+1}: Simplified to {len(mesh.faces)} faces")
                except Exception as simp_err:
                    print(f"[{job_id}] Layer {i+1}: Simplification failed: {simp_err}")

            # Apply vertex colors
            vertex_colors = np.tile(color, (len(mesh.vertices), 1)).astype(np.uint8)
            mesh.visual.vertex_colors = vertex_colors

            # Store region name
            mesh.metadata["region_name"] = f"layer_{i}"

            meshes.append(mesh)
            print(f"[{job_id}] Layer {i+1}: Added to meshes")

        except Exception as layer_err:
            print(f"[{job_id}] Layer {i+1} failed: {layer_err}")
            continue

    return meshes


def _generate_atlas_region_meshes(
    job_id: str,
    cortical_labels: np.ndarray,
    subcortical_labels: np.ndarray,
    spacing: tuple,
    progress_callback=None
) -> tuple[list, list]:
    """Generate meshes for atlas-defined brain regions."""
    meshes = []
    regions_metadata = []

    atlas_regions = [
        ("ventricles", subcortical_labels, REGION_DEFINITIONS["ventricles"]),
        ("frontal_lobe", cortical_labels, REGION_DEFINITIONS["frontal_lobe"]),
        ("temporal_lobe", cortical_labels, REGION_DEFINITIONS["temporal_lobe"]),
        ("parietal_lobe", cortical_labels, REGION_DEFINITIONS["parietal_lobe"]),
        ("occipital_lobe", cortical_labels, REGION_DEFINITIONS["occipital_lobe"]),
    ]

    for i, (region_name, atlas_data, region_def) in enumerate(atlas_regions):
        if progress_callback:
            progress_callback(75 + (i * 3))

        print(f"[{job_id}] Processing atlas region: {region_name}...")

        if "atlas_labels" not in region_def:
            continue

        mask = create_region_mask(atlas_data, region_def["atlas_labels"])

        if np.sum(mask) == 0:
            print(f"[{job_id}] {region_name}: No voxels found, skipping")
            continue

        # Use more transparent colors for atlas overlay
        color = region_def["color"].copy()
        color[3] = 100  # Make more transparent for overlay

        mesh = mask_to_mesh(
            mask,
            spacing,
            color,
            job_id,
            region_name,
            max_faces=100000,
            smoothing_iterations=5
        )

        if mesh:
            meshes.append(mesh)
            regions_metadata.append({
                "name": region_name,
                "label": region_def["label"],
                "color": region_def["color"][:3],
                "opacity": 0.3,  # Lower opacity for atlas overlay
                "defaultVisible": False,  # Hidden by default, user can toggle
            })

    return meshes, regions_metadata


def process_with_atlas(
    job_id: str,
    patient_mri_path: str,
    patient_data: np.ndarray,
    spacing: tuple,
    is_segmentation: bool = False,
    progress_callback=None
) -> tuple[list[trimesh.Trimesh], dict]:
    """
    Process patient MRI with atlas-based segmentation.
    If input is already a segmentation, process labels directly without atlas registration.

    Returns:
        Tuple of (list of meshes, metadata dict)
    """
    meshes = []
    regions_metadata = []
    has_tumor = False
    atlas_registered = False

    if is_segmentation:
        # Input is already a segmentation - process labels AND apply atlas
        print(f"[{job_id}] Input is segmentation data - processing labels and applying atlas")
        meshes, regions_metadata, has_tumor = _process_segmentation_labels(
            job_id, patient_data, spacing, progress_callback
        )

        # Also try to apply atlas regions for anatomical context
        if progress_callback:
            progress_callback(70)

        cortical_labels, subcortical_labels = register_atlas_to_patient(patient_mri_path, job_id)
        atlas_registered = cortical_labels is not None and subcortical_labels is not None

        if atlas_registered:
            print(f"[{job_id}] Adding atlas regions for anatomical context...")
            atlas_meshes, atlas_metadata = _generate_atlas_region_meshes(
                job_id, cortical_labels, subcortical_labels, spacing, progress_callback
            )
            meshes.extend(atlas_meshes)
            regions_metadata.extend(atlas_metadata)
    else:
        # Input is intensity MRI - use atlas registration
        # Step 1: Generate brain shell from intensity data
        if progress_callback:
            progress_callback(25)
        print(f"[{job_id}] Generating brain shell...")

        brain_shell = generate_brain_shell_mesh(patient_data, spacing, job_id)
        if brain_shell:
            meshes.append(brain_shell)
            regions_metadata.append({
                "name": "brain_shell",
                "label": REGION_DEFINITIONS["brain_shell"]["label"],
                "color": REGION_DEFINITIONS["brain_shell"]["color"][:3],
                "opacity": REGION_DEFINITIONS["brain_shell"]["opacity"],
                "defaultVisible": True,
            })

        # Step 2: Register atlas to patient space
        if progress_callback:
            progress_callback(30)

        cortical_labels, subcortical_labels = register_atlas_to_patient(patient_mri_path, job_id)

        atlas_registered = cortical_labels is not None and subcortical_labels is not None

        if atlas_registered:
            # Step 3: Generate meshes for each atlas region
            atlas_regions = [
                ("ventricles", subcortical_labels, REGION_DEFINITIONS["ventricles"]),
                ("frontal_lobe", cortical_labels, REGION_DEFINITIONS["frontal_lobe"]),
                ("temporal_lobe", cortical_labels, REGION_DEFINITIONS["temporal_lobe"]),
                ("parietal_lobe", cortical_labels, REGION_DEFINITIONS["parietal_lobe"]),
                ("occipital_lobe", cortical_labels, REGION_DEFINITIONS["occipital_lobe"]),
            ]

            for i, (region_name, atlas_data, region_def) in enumerate(atlas_regions):
                if progress_callback:
                    progress_callback(40 + (i * 10))

                print(f"[{job_id}] Processing {region_name}...")

                if "atlas_labels" not in region_def:
                    continue

                mask = create_region_mask(atlas_data, region_def["atlas_labels"])

                if np.sum(mask) == 0:
                    print(f"[{job_id}] {region_name}: No voxels found, skipping")
                    continue

                mesh = mask_to_mesh(
                    mask,
                    spacing,
                    region_def["color"],
                    job_id,
                    region_name
                )

                if mesh:
                    meshes.append(mesh)
                    regions_metadata.append({
                        "name": region_name,
                        "label": region_def["label"],
                        "color": region_def["color"][:3],
                        "opacity": region_def["opacity"],
                        "defaultVisible": region_def["defaultVisible"],
                    })
        else:
            print(f"[{job_id}] Atlas registration failed, generating intensity-based layers instead")
            # Fallback: generate intensity-based layers directly
            intensity_meshes = _generate_intensity_layers(job_id, patient_data, spacing, progress_callback)
            meshes.extend(intensity_meshes)
            for i, m in enumerate(intensity_meshes):
                regions_metadata.append({
                    "name": f"layer_{i}",
                    "label": f"Layer {i+1}",
                    "color": [150, 150, 150],
                    "opacity": 0.7,
                    "defaultVisible": True,
                })

    if progress_callback:
        progress_callback(90)

    metadata = {
        "regions": regions_metadata,
        "has_tumor": has_tumor,
        "atlas_registered": atlas_registered,
        "input_type": "segmentation" if is_segmentation else "intensity",
    }

    return meshes, metadata


def _process_segmentation_labels(
    job_id: str,
    seg_data: np.ndarray,
    spacing: tuple,
    progress_callback=None
) -> tuple[list, list, bool]:
    """Process segmentation labels directly into colored meshes."""
    meshes = []
    regions_metadata = []

    # Get unique labels (excluding background 0)
    unique_labels = np.unique(seg_data[seg_data > 0]).astype(int).tolist()
    print(f"[{job_id}] Found {len(unique_labels)} segmentation labels: {unique_labels}")

    # Color palette for segmentation labels
    label_colors = {
        1: ([220, 180, 180, 255], "Region 1"),
        2: ([180, 220, 180, 255], "Region 2"),
        3: ([180, 180, 220, 255], "Region 3"),
        4: ([255, 200, 100, 255], "Region 4"),
        5: ([200, 100, 200, 255], "Region 5"),
        6: ([100, 200, 200, 255], "Region 6"),
        7: ([255, 150, 150, 255], "Region 7"),
        8: ([150, 255, 150, 255], "Region 8"),
        9: ([150, 150, 255, 255], "Region 9"),
        10: ([255, 255, 150, 255], "Region 10"),
    }
    default_color = [200, 200, 200, 255]

    for i, label in enumerate(unique_labels):
        if progress_callback:
            progress_callback(25 + int((i / len(unique_labels)) * 60))

        print(f"[{job_id}] Processing label {label} ({i+1}/{len(unique_labels)})")

        mask = (seg_data == label).astype(np.float32)
        color_info = label_colors.get(int(label), (default_color, f"Region {label}"))
        color = color_info[0]
        label_name = color_info[1]

        region_name = f"label_{label}"

        mesh = mask_to_mesh(
            mask,
            spacing,
            color,
            job_id,
            region_name,
            max_faces=150000,
            smoothing_iterations=5
        )

        if mesh:
            meshes.append(mesh)
            regions_metadata.append({
                "name": region_name,
                "label": label_name,
                "color": color[:3],
                "opacity": 1.0,
                "defaultVisible": True,
            })

    # Check for tumor (only if it looks like BraTS format)
    has_tumor = False
    tumor_mask = detect_tumor_from_segmentation(seg_data, unique_labels)
    if tumor_mask is not None:
        print(f"[{job_id}] Detected BraTS-style tumor segmentation")
        has_tumor = True
        # Note: tumor regions are already included as individual labels above

    return meshes, regions_metadata, has_tumor


def export_meshes_with_metadata(
    meshes: list[trimesh.Trimesh],
    metadata: dict,
    output_path: Path,
    metadata_path: Path,
    job_id: str
) -> None:
    """Export meshes to GLB and save metadata to JSON."""
    if not meshes:
        raise ValueError("No meshes to export")

    print(f"[{job_id}] Combining {len(meshes)} meshes...")

    # Create a scene with named meshes
    scene = trimesh.Scene()
    for i, mesh in enumerate(meshes):
        region_name = mesh.metadata.get("region_name", f"mesh_{i}")
        scene.add_geometry(mesh, node_name=region_name)

    print(f"[{job_id}] Exporting scene to GLB...")
    scene.export(str(output_path), file_type="glb")

    print(f"[{job_id}] Saving metadata to JSON...")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[{job_id}] Export complete: {output_path}")
