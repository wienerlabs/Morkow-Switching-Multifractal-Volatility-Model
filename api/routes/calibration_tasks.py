"""Status polling endpoint for async calibration tasks."""

from fastapi import APIRouter, HTTPException

from api.models import TaskStatusResponse
from api.tasks import get_task

router = APIRouter(tags=["calibration-tasks"])


@router.get("/calibrate/status/{task_id}", response_model=TaskStatusResponse, summary="Poll task status")
def get_task_status(task_id: str):
    """Poll the status of an async calibration task. Returns result when complete."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return {
        "task_id": task["task_id"],
        "status": task["status"].value if hasattr(task["status"], "value") else task["status"],
        "endpoint": task["endpoint"],
        "result": task["result"],
        "error": task["error"],
        "created_at": task["created_at"],
        "started_at": task["started_at"],
        "completed_at": task["completed_at"],
    }

