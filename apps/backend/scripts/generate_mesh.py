#!/usr/bin/env python3
"""Generate OBJ mesh from NIfTI brain scan using tumor-first pipeline.

Pipeline:
1. Run MONAI tumor detection (requires 4 modalities: T1, T1ce, T2, FLAIR)
2. Mask out tumor regions from T1
3. Run ANTsPyNet deep_atropos on masked brain
4. Combine tumor + anatomical segmentations
5. Generate OBJ mesh
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

from services.synthseg_inference import run_synthseg
from services.tumor_inference import run_tumor_detection, TumorInference
from services.synthseg_service import get_label_info, remap_brats_labels


def run_pipeline(
    t1_path: Path,
    output_dir: Path,
    t1ce_path: Path = None,
    t2_path: Path = None,
    flair_path: Path = None,
    job_id: str = "cli"
):
    """
    Run full pipeline: tumor detection -> masked segmentation -> mesh generation.

    If all 4 modalities provided: runs tumor detection first, masks tumor, then anatomical.
    If only T1: runs anatomical segmentation only.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    tumor_seg_path = output_dir / f"{job_id}_tumor.nii.gz"
    anatomical_seg_path = output_dir / f"{job_id}_anatomical.nii.gz"
    combined_seg_path = output_dir / f"{job_id}_combined.nii.gz"
    obj_path = output_dir / f"{job_id}_brain.obj"
    metadata_path = output_dir / f"{job_id}_metadata.json"

    has_multimodal = all([t1ce_path, t2_path, flair_path])
    tumor_seg = None

    # Step 1: Run tumor detection if we have all modalities
    if has_multimodal:
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
            t1ce_path,  # Reference for affine
            progress_callback=lambda p: print(f"  Tumor detection: {p}%")
        )

        if tumor_seg is not None and np.any(tumor_seg > 0):
            tumor_voxels = np.sum(tumor_seg > 0)
            print(f"Tumor detected: {tumor_voxels} voxels")
        else:
            print("No tumor detected")
            tumor_seg = None
    else:
        print(f"\n{'='*60}")
        print("Step 1: Skipping tumor detection (need all 4 modalities)")
        print(f"{'='*60}")
        print(f"  T1: {t1_path}")
        print(f"  T1ce: {t1ce_path}")
        print(f"  T2: {t2_path}")
        print(f"  FLAIR: {flair_path}")

    # Step 2: Mask out tumor from T1 if tumor was found
    print(f"\n{'='*60}")
    print("Step 2: Preparing brain for anatomical segmentation...")
    print(f"{'='*60}")

    if tumor_seg is not None and np.any(tumor_seg > 0):
        print("Masking out tumor regions from T1...")

        # Load T1
        t1_img = nib.load(str(t1_path))
        t1_data = t1_img.get_fdata()

        # Create tumor mask (any tumor label > 0)
        # tumor_seg uses remapped labels: 100=necrotic, 101=edema, 102=enhancing
        tumor_mask = tumor_seg > 0

        # Dilate tumor mask slightly to ensure clean boundaries
        from scipy.ndimage import binary_dilation
        tumor_mask_dilated = binary_dilation(tumor_mask, iterations=2)

        # Fill tumor region with surrounding tissue intensity (inpainting)
        # Simple approach: use median of non-tumor brain tissue
        brain_mask = t1_data > np.percentile(t1_data[t1_data > 0], 10)
        non_tumor_brain = t1_data[brain_mask & ~tumor_mask_dilated]
        fill_value = np.median(non_tumor_brain) if len(non_tumor_brain) > 0 else 0

        t1_masked = t1_data.copy()
        t1_masked[tumor_mask_dilated] = fill_value

        # Save masked T1 for processing
        masked_t1_path = output_dir / f"{job_id}_t1_masked.nii.gz"
        masked_img = nib.Nifti1Image(t1_masked.astype(np.float32), t1_img.affine)
        nib.save(masked_img, str(masked_t1_path))

        input_for_synthseg = masked_t1_path
        print(f"Masked T1 saved to: {masked_t1_path}")
    else:
        input_for_synthseg = t1_path
        print("No tumor to mask, using original T1")

    # Step 3: Run anatomical segmentation
    print(f"\n{'='*60}")
    print("Step 3: Running anatomical segmentation (ANTsPyNet deep_atropos)...")
    print(f"{'='*60}")

    anatomical_seg = run_synthseg(
        input_path=input_for_synthseg,
        output_path=anatomical_seg_path,
        flair_path=flair_path if has_multimodal else None,
        t2_path=t2_path if has_multimodal else None,
        progress_callback=lambda p: print(f"  Anatomical segmentation: {p}%")
    )

    print(f"Anatomical segmentation saved to: {anatomical_seg_path}")

    # Step 4: Combine segmentations
    print(f"\n{'='*60}")
    print("Step 4: Combining segmentations...")
    print(f"{'='*60}")

    if tumor_seg is not None and np.any(tumor_seg > 0):
        # Combine: anatomical as base, tumor overlaid
        combined = anatomical_seg.copy()

        # Overlay tumor labels (they take precedence)
        tumor_mask = tumor_seg > 0
        combined[tumor_mask] = tumor_seg[tumor_mask]

        # Save combined
        ref_img = nib.load(str(t1_path))
        combined_img = nib.Nifti1Image(combined.astype(np.int16), ref_img.affine)
        nib.save(combined_img, str(combined_seg_path))

        final_seg_path = combined_seg_path
        print(f"Combined segmentation saved to: {combined_seg_path}")
    else:
        final_seg_path = anatomical_seg_path
        combined = anatomical_seg
        print("No tumor, using anatomical segmentation only")

    # Step 5: Generate mesh
    print(f"\n{'='*60}")
    print("Step 5: Generating 3D mesh...")
    print(f"{'='*60}")

    img = nib.load(str(final_seg_path))
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
    print(f"\n{'='*60}")
    print("Step 6: Exporting to OBJ format...")
    print(f"{'='*60}")

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
    has_tumor = tumor_seg is not None and np.any(tumor_seg > 0)
    metadata = {
        "regions": regions_metadata,
        "has_tumor": has_tumor,
        "total_regions": len(regions_metadata),
        "segmentation_method": "tumor_first_pipeline",
        "files": {
            "combined_obj": str(obj_path),
            "combined_segmentation": str(final_seg_path),
            "anatomical_segmentation": str(anatomical_seg_path),
            "tumor_segmentation": str(tumor_seg_path) if has_tumor else None,
            "individual_regions": str(individual_dir)
        }
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")
    print(f"  Combined OBJ: {obj_path}")
    print(f"  Segmentation: {final_seg_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Tumor found: {has_tumor}")

    return obj_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate OBJ mesh from brain MRI using tumor-first pipeline"
    )
    parser.add_argument("t1", type=Path, help="T1 NIfTI file (.nii or .nii.gz)")
    parser.add_argument("--t1ce", type=Path, help="T1ce (contrast-enhanced) NIfTI")
    parser.add_argument("--t2", type=Path, help="T2 NIfTI")
    parser.add_argument("--flair", type=Path, help="FLAIR NIfTI")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--job-id", type=str, default="brain", help="Job ID for output files")

    args = parser.parse_args()

    if not args.t1.exists():
        print(f"Error: T1 file not found: {args.t1}")
        sys.exit(1)

    # Validate optional modalities
    for name, path in [("t1ce", args.t1ce), ("t2", args.t2), ("flair", args.flair)]:
        if path and not path.exists():
            print(f"Error: {name} file not found: {path}")
            sys.exit(1)

    output_dir = args.output or args.t1.parent / "output"

    run_pipeline(
        t1_path=args.t1,
        output_dir=output_dir,
        t1ce_path=args.t1ce,
        t2_path=args.t2,
        flair_path=args.flair,
        job_id=args.job_id
    )


if __name__ == "__main__":
    main()
