#!/usr/bin/env python3
"""Generate OBJ mesh from NIfTI brain scan.

Modes:
1. Multi-modal (T1, T1ce, T2, FLAIR): Runs MONAI tumor detection, generates mesh
2. Pre-segmented NIfTI: Directly generates mesh from segmentation labels
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
import trimesh

from services.tumor_inference import run_tumor_detection
from services.synthseg_service import get_label_info


def generate_mesh_from_segmentation(
    segmentation_path: Path,
    output_dir: Path,
    job_id: str = "cli"
):
    """Generate OBJ mesh from a pre-segmented NIfTI file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    obj_path = output_dir / f"{job_id}_brain.obj"
    metadata_path = output_dir / f"{job_id}_metadata.json"

    print(f"\n{'='*60}")
    print("Generating 3D mesh from segmentation...")
    print(f"{'='*60}")

    img = nib.load(str(segmentation_path))
    data = img.get_fdata().astype(np.int32)
    spacing = img.header.get_zooms()[:3]

    unique_labels = np.unique(data[data > 0]).astype(int).tolist()
    print(f"Found {len(unique_labels)} regions: {unique_labels}")

    meshes = []
    regions_metadata = []

    for i, label_id in enumerate(unique_labels):
        label_info = get_label_info(label_id)
        region_name = label_info["name"]
        color = label_info["color"]
        category = label_info["category"]

        print(f"  Processing {region_name} (label {label_id})...")

        mask = (data == label_id).astype(np.float32)
        mask_smoothed = gaussian_filter(mask, sigma=0.5)

        try:
            verts, faces, normals, _ = marching_cubes(
                mask_smoothed,
                level=0.5,
                spacing=spacing
            )

            if len(verts) == 0:
                print(f"    {region_name}: No vertices, skipping")
                continue

            mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces,
                vertex_normals=normals
            )

            trimesh.smoothing.filter_laplacian(mesh, iterations=5)

            max_faces = 100000
            if len(mesh.faces) > max_faces:
                try:
                    mesh = mesh.simplify_quadric_decimation(max_faces)
                except Exception:
                    pass

            mesh.metadata["region_name"] = region_name
            meshes.append(mesh)

            is_tumor = category == "tumor"
            is_outer = category in ["cortex", "white_matter", "cerebellum"]

            regions_metadata.append({
                "name": region_name,
                "label": region_name.replace("_", " ").title(),
                "labelId": label_id,
                "color": color,
                "category": category,
                "opacity": 1.0 if is_tumor else (0.3 if is_outer else 0.8),
                "defaultVisible": True,
                "vertexCount": len(mesh.vertices),
                "faceCount": len(mesh.faces),
            })

            print(f"    {region_name}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        except Exception as e:
            print(f"    {region_name} failed: {e}")
            continue

    if not meshes:
        raise ValueError("No meshes could be generated")

    # Export combined OBJ
    print(f"\nExporting to OBJ format...")

    scene = trimesh.Scene()
    for mesh in meshes:
        region_name = mesh.metadata.get("region_name", "unknown")
        scene.add_geometry(mesh, node_name=region_name)

    scene.export(str(obj_path), file_type="obj")
    print(f"OBJ saved to: {obj_path}")

    # Export individual regions
    individual_dir = output_dir / "regions"
    individual_dir.mkdir(exist_ok=True)

    for mesh in meshes:
        region_name = mesh.metadata.get("region_name", "unknown")
        region_obj = individual_dir / f"{region_name}.obj"
        mesh.export(str(region_obj), file_type="obj")

    print(f"Individual region OBJs saved to: {individual_dir}")

    # Save metadata
    has_tumor = any(r["category"] == "tumor" for r in regions_metadata)
    metadata = {
        "regions": regions_metadata,
        "has_tumor": has_tumor,
        "total_regions": len(regions_metadata),
        "segmentation_method": "pre_segmented",
        "files": {
            "combined_obj": str(obj_path),
            "individual_regions": str(individual_dir)
        }
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("Mesh generation complete!")
    print(f"{'='*60}")
    print(f"  OBJ: {obj_path}")
    print(f"  Metadata: {metadata_path}")

    return obj_path


def run_tumor_pipeline(
    t1_path: Path,
    t1ce_path: Path,
    t2_path: Path,
    flair_path: Path,
    output_dir: Path,
    job_id: str = "cli"
):
    """Run MONAI tumor detection on multi-modal MRI and generate mesh."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tumor_seg_path = output_dir / f"{job_id}_tumor.nii.gz"
    obj_path = output_dir / f"{job_id}_tumor.obj"
    metadata_path = output_dir / f"{job_id}_metadata.json"

    print(f"\n{'='*60}")
    print("Step 1: Running tumor detection (MONAI SegResNet)...")
    print(f"{'='*60}")

    modality_paths = {
        "t1": t1_path,
        "t1ce": t1ce_path,
        "t2": t2_path,
        "flair": flair_path
    }

    tumor_seg = run_tumor_detection(
        modality_paths,
        tumor_seg_path,
        t1ce_path,
        progress_callback=lambda p: print(f"  Tumor detection: {p}%")
    )

    if tumor_seg is None or not np.any(tumor_seg > 0):
        print("No tumor detected in scan")
        metadata = {
            "regions": [],
            "has_tumor": False,
            "total_regions": 0,
            "segmentation_method": "monai_tumor_detection",
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return None

    tumor_voxels = np.sum(tumor_seg > 0)
    print(f"Tumor detected: {tumor_voxels} voxels")

    # Generate mesh from tumor segmentation
    print(f"\n{'='*60}")
    print("Step 2: Generating 3D mesh from tumor segmentation...")
    print(f"{'='*60}")

    return generate_mesh_from_segmentation(tumor_seg_path, output_dir, job_id)


def main():
    parser = argparse.ArgumentParser(
        description="Generate OBJ mesh from brain MRI segmentation or run tumor detection"
    )
    parser.add_argument("input", type=Path, help="Input NIfTI file (segmentation or T1 for tumor mode)")
    parser.add_argument("--t1ce", type=Path, help="T1ce for tumor detection mode")
    parser.add_argument("--t2", type=Path, help="T2 for tumor detection mode")
    parser.add_argument("--flair", type=Path, help="FLAIR for tumor detection mode")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--job-id", type=str, default="brain", help="Job ID for output files")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    output_dir = args.output or args.input.parent / "output"

    # Check if running in tumor detection mode (all 4 modalities provided)
    has_multimodal = all([args.t1ce, args.t2, args.flair])

    if has_multimodal:
        # Validate all files exist
        for name, path in [("t1ce", args.t1ce), ("t2", args.t2), ("flair", args.flair)]:
            if not path.exists():
                print(f"Error: {name} file not found: {path}")
                sys.exit(1)

        run_tumor_pipeline(
            t1_path=args.input,
            t1ce_path=args.t1ce,
            t2_path=args.t2,
            flair_path=args.flair,
            output_dir=output_dir,
            job_id=args.job_id
        )
    else:
        # Treat input as pre-segmented NIfTI
        generate_mesh_from_segmentation(
            segmentation_path=args.input,
            output_dir=output_dir,
            job_id=args.job_id
        )


if __name__ == "__main__":
    main()
