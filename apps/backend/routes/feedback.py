"""AI-powered case feedback API endpoints."""
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException

from database import Database
from models.feedback import (
    FeedbackMessageCreate,
    FeedbackMessageResponse,
    FeedbackSessionResponse,
    GenerateSummaryResponse,
    ChatResponse,
)
from services.gemini_service import generate_summary, chat_response
from services.metrics_service import (
    calculate_tumor_metrics_from_segmentation,
    get_scan_regions_from_metadata,
    aggregate_case_metrics,
)

router = APIRouter()

# Paths
BASE_DIR = Path(__file__).parent.parent
SEGMENTATION_DIR = BASE_DIR / "storage" / "segmentations"
METADATA_DIR = BASE_DIR / "storage" / "metadata"


async def get_or_create_session(patient_id: int, case_id: int) -> dict:
    """Get existing session or create a new one."""
    session = await Database.feedback_sessions.find_one({
        "patient_id": patient_id,
        "case_id": case_id
    })

    if not session:
        session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "patient_id": patient_id,
            "case_id": case_id,
            "summary": None,
            "metrics": None,
            "scan_count": 0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        await Database.feedback_sessions.insert_one(session)

    return session


async def get_case_scan_data(patient_id: int, case_id: int) -> List[dict]:
    """Get all scan data for a case including metrics."""
    cursor = Database.scan_files.find({
        "patient_id": patient_id,
        "case_id": case_id,
        "status": "completed"
    }).sort("scan_timestamp", 1)

    files = await cursor.to_list(length=None)
    scan_data = []

    for file_doc in files:
        job_id = file_doc["job_id"]

        # Try to get metrics from segmentation
        seg_path = SEGMENTATION_DIR / f"{job_id}_seg.nii.gz"
        metrics = None
        if seg_path.exists():
            metrics = calculate_tumor_metrics_from_segmentation(seg_path)

        # Get regions from metadata
        metadata_path = METADATA_DIR / f"{job_id}.json"
        regions = get_scan_regions_from_metadata(metadata_path)

        scan_data.append({
            "job_id": job_id,
            "filename": file_doc.get("original_file", {}).get("filename", "Unknown"),
            "scan_date": file_doc.get("scan_timestamp", file_doc.get("uploaded_at")),
            "has_tumor": metrics.get("has_tumor", False) if metrics else False,
            "metrics": metrics,
            "regions_detected": regions
        })

    return scan_data


@router.get("/{patient_id}/{case_id}", response_model=FeedbackSessionResponse)
async def get_feedback_session(patient_id: int, case_id: int):
    """Get feedback session with chat history."""
    try:
        session = await get_or_create_session(patient_id, case_id)

        # Get messages
        cursor = Database.feedback_messages.find({
            "session_id": session["session_id"]
        }).sort("created_at", 1)

        messages = await cursor.to_list(length=None)

        return FeedbackSessionResponse(
            session_id=session["session_id"],
            patient_id=session["patient_id"],
            case_id=session["case_id"],
            summary=session.get("summary"),
            metrics=session.get("metrics"),
            scan_count=session.get("scan_count", 0),
            messages=[
                FeedbackMessageResponse(
                    message_id=m["message_id"],
                    role=m["role"],
                    content=m["content"],
                    created_at=m["created_at"]
                )
                for m in messages
            ],
            created_at=session["created_at"],
            updated_at=session["updated_at"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feedback session: {str(e)}")


@router.post("/{patient_id}/{case_id}/generate", response_model=GenerateSummaryResponse)
async def generate_case_summary(patient_id: int, case_id: int):
    """Generate a new AI summary for the case."""
    try:
        # Get or create session
        session = await get_or_create_session(patient_id, case_id)

        # Get scan data
        scan_data = await get_case_scan_data(patient_id, case_id)

        if not scan_data:
            raise HTTPException(
                status_code=404,
                detail="No completed scans found for this case"
            )

        # Calculate aggregated metrics
        scan_metrics = [s.get("metrics", {}) for s in scan_data if s.get("metrics")]
        aggregated_metrics = aggregate_case_metrics(scan_metrics)

        # Generate summary with Gemini
        summary = await generate_summary(
            patient_id=patient_id,
            case_id=case_id,
            scan_summaries=scan_data,
            aggregated_metrics=aggregated_metrics
        )

        # Update session
        await Database.feedback_sessions.update_one(
            {"session_id": session["session_id"]},
            {
                "$set": {
                    "summary": summary,
                    "metrics": aggregated_metrics,
                    "scan_count": len(scan_data),
                    "updated_at": datetime.utcnow()
                }
            }
        )

        # Add summary as system message
        message_id = str(uuid.uuid4())
        await Database.feedback_messages.insert_one({
            "message_id": message_id,
            "session_id": session["session_id"],
            "patient_id": patient_id,
            "case_id": case_id,
            "role": "assistant",
            "content": summary,
            "created_at": datetime.utcnow()
        })

        return GenerateSummaryResponse(
            session_id=session["session_id"],
            summary=summary,
            metrics=aggregated_metrics,
            message="Summary generated successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@router.post("/{patient_id}/{case_id}/chat", response_model=ChatResponse)
async def send_chat_message(
    patient_id: int,
    case_id: int,
    message: FeedbackMessageCreate
):
    """Send a chat message and get AI response."""
    try:
        session = await get_or_create_session(patient_id, case_id)

        # Save user message
        user_message_id = str(uuid.uuid4())
        user_message_doc = {
            "message_id": user_message_id,
            "session_id": session["session_id"],
            "patient_id": patient_id,
            "case_id": case_id,
            "role": "user",
            "content": message.content,
            "created_at": datetime.utcnow()
        }
        await Database.feedback_messages.insert_one(user_message_doc)

        # Get conversation history
        cursor = Database.feedback_messages.find({
            "session_id": session["session_id"]
        }).sort("created_at", 1)
        history = await cursor.to_list(length=None)

        # Build messages for Gemini
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in history
        ]

        # Get case context
        case_context = {
            "patient_id": patient_id,
            "case_id": case_id,
            "scan_count": session.get("scan_count", 0),
            "has_tumor": session.get("metrics", {}).get("has_tumor", False),
            "metrics": session.get("metrics")
        }

        # Get AI response
        ai_response = await chat_response(messages, case_context)

        # Save AI response
        ai_message_id = str(uuid.uuid4())
        ai_message_doc = {
            "message_id": ai_message_id,
            "session_id": session["session_id"],
            "patient_id": patient_id,
            "case_id": case_id,
            "role": "assistant",
            "content": ai_response,
            "created_at": datetime.utcnow()
        }
        await Database.feedback_messages.insert_one(ai_message_doc)

        # Update session timestamp
        await Database.feedback_sessions.update_one(
            {"session_id": session["session_id"]},
            {"$set": {"updated_at": datetime.utcnow()}}
        )

        return ChatResponse(
            user_message=FeedbackMessageResponse(
                message_id=user_message_id,
                role="user",
                content=message.content,
                created_at=user_message_doc["created_at"]
            ),
            assistant_message=FeedbackMessageResponse(
                message_id=ai_message_id,
                role="assistant",
                content=ai_response,
                created_at=ai_message_doc["created_at"]
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process chat message: {str(e)}")


@router.delete("/{patient_id}/{case_id}", status_code=204)
async def clear_feedback_session(patient_id: int, case_id: int):
    """Clear chat history and summary for a case."""
    try:
        session = await Database.feedback_sessions.find_one({
            "patient_id": patient_id,
            "case_id": case_id
        })

        if session:
            # Delete all messages
            await Database.feedback_messages.delete_many({
                "session_id": session["session_id"]
            })

            # Reset session
            await Database.feedback_sessions.update_one(
                {"session_id": session["session_id"]},
                {
                    "$set": {
                        "summary": None,
                        "metrics": None,
                        "updated_at": datetime.utcnow()
                    }
                }
            )

        return None

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")
