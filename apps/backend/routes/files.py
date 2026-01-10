"""Scan file management API endpoints."""
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path

from database import Database
from models.scan_file import ScanFileResponse

router = APIRouter()

# Path to mesh storage
MESH_DIR = Path(__file__).parent.parent / "storage" / "meshes"


@router.get("/{patient_id}/{case_id}", response_model=List[ScanFileResponse])
async def list_files(patient_id: int, case_id: int):
    """List all scan files for a case."""
    try:
        # Query files for this case
        cursor = Database.scan_files.find({
            "patient_id": patient_id,
            "case_id": case_id
        }).sort("uploaded_at", -1)

        files = await cursor.to_list(length=None)

        # Convert to response format
        result = []
        for file_doc in files:
            result.append(ScanFileResponse(
                job_id=file_doc["job_id"],
                file_id=file_doc["file_id"],
                case_id=file_doc["case_id"],
                patient_id=file_doc["patient_id"],
                original_filename=file_doc["original_file"]["filename"],
                status=file_doc["status"],
                progress=file_doc["progress"],
                mesh_url=f"/api/mesh/{file_doc['job_id']}" if file_doc["status"] == "completed" else None,
                original_url=None,
                error=file_doc.get("error"),
                uploaded_at=file_doc["uploaded_at"],
                scan_timestamp=file_doc["scan_timestamp"],
                doctor_notes=file_doc.get("doctor_notes"),
                metadata=file_doc.get("metadata", {})
            ))

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@router.delete("/{patient_id}/{case_id}/{job_id}", status_code=204)
async def delete_file(patient_id: int, case_id: int, job_id: str):
    """Delete a scan file."""
    try:
        # Find the file
        file_doc = await Database.scan_files.find_one({
            "patient_id": patient_id,
            "case_id": case_id,
            "job_id": job_id
        })

        if not file_doc:
            raise HTTPException(
                status_code=404,
                detail=f"File '{job_id}' not found for case {case_id} and patient {patient_id}"
            )

        # Delete the mesh file from filesystem if it exists
        mesh_path = MESH_DIR / f"{job_id}.glb"
        if mesh_path.exists():
            mesh_path.unlink()

        # Delete the document from database
        result = await Database.scan_files.delete_one({
            "patient_id": patient_id,
            "case_id": case_id,
            "job_id": job_id
        })

        if result.deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"File '{job_id}' not found"
            )

        return None

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")
