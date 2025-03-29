import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Enhanced Study Assistant API", description="Track study sessions with improved focus detection")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Mount static files directory
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# In-memory storage for study sessions
# In a production app, you would use a proper database
study_sessions = {}

# Pydantic models for data validation
class BlinkMetrics(BaseModel):
    blink_count: int
    blink_rate: float  # Blinks per minute
    avg_blink_duration: Optional[float] = None  # In milliseconds

class GazeMetrics(BaseModel):
    blink_rate: Optional[float] = None
    attention_score: Optional[float] = None
    looking_at_content_percentage: Optional[float] = None
    looking_at_screen_percentage: Optional[float] = None

class ClientMetrics(BaseModel):
    timestamp: str
    blink_metrics: BlinkMetrics
    seconds_since_last_movement: Optional[int] = None
    client_estimated_attention: Optional[float] = None  # 0-1 scale
    tracking_mode: Optional[str] = None  # 'gaze', 'face', 'motion', 'memory', 'inactive'
    motion_level: Optional[float] = None  # Motion level detected
    input_events: Optional[int] = None  # Keyboard/mouse event count
    gaze_metrics: Optional[GazeMetrics] = None  # New gaze tracking metrics

@app.get("/")
def read_root():
    return {"message": "Enhanced Study Assistant API is running"}

# Add this to your app.py file to explicitly create the static/models directory
# Place this right after the existing static_dir check

# Ensure models directory exists
models_dir = os.path.join(static_dir, "models")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Created directory: {models_dir}")

# Add a route to check if face models are available
@app.get("/check-models")
def check_models():
    """
    Check if face detection model files are available
    """
    models_dir = os.path.join("static", "models")
    required_files = [
        "tiny_face_detector_model-weights_manifest.json",
        "tiny_face_detector_model-shard1",
        "face_landmark_68_model-weights_manifest.json",
        "face_landmark_68_model-shard1"
    ]
    
    # Check each required file
    missing_files = []
    for file in required_files:
        file_path = os.path.join(models_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        return {
            "status": "missing_files",
            "missing": missing_files,
            "message": "Some required face model files are missing. Run download_face_models.py to download them."
        }
    else:
        return {
            "status": "ok",
            "message": "All required face model files are available."
        }

@app.get("/client", response_class=HTMLResponse)
async def get_client():
    """
    Serve the enhanced client HTML page directly from FastAPI.
    """
    try:
        with open("enhanced_client.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        # Fallback to test_client if enhanced version isn't found
        try:
            with open("test_client.html", "r") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        except FileNotFoundError:
            return HTMLResponse(content="<html><body><h1>Client file not found</h1><p>Please ensure the client HTML file exists.</p></body></html>")

@app.post("/start-session/")
def start_session():
    """
    Start a new study session and generate a session ID.
    """
    session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    study_sessions[session_id] = {
        "start_time": datetime.now().isoformat(),
        "frames": [],
        "client_metrics": [],
        "metrics": {
            "blink_rate": [],
            "posture_changes": [],
            "distraction_events": [],
            "focus_periods": [],
            "tracking_modes": []  # New metric to track when different tracking modes are used
        },
        "active_times": [],  # Track active study periods (start, end) pairs
        "current_active_start": datetime.now().isoformat(),  # Track when current active period started
        "paused_times": [],  # Track paused periods (start, end) pairs
        "end_time": None
    }
    return {"session_id": session_id, "message": "Study session started"}

@app.post("/pause-session/{session_id}")
def pause_session(session_id: str):
    """
    Pause a study session to track breaks.
    """
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    now = datetime.now().isoformat()
    
    # Record the end of the current active period
    if "current_active_start" in study_sessions[session_id]:
        active_start = study_sessions[session_id]["current_active_start"]
        study_sessions[session_id]["active_times"].append({
            "start": active_start,
            "end": now
        })
        study_sessions[session_id]["current_active_start"] = None
    
    # Start a new pause period
    study_sessions[session_id]["paused_times"].append({
        "start": now,
        "end": None
    })
    
    return {"message": "Session paused", "timestamp": now}

@app.post("/resume-session/{session_id}")
def resume_session(session_id: str):
    """
    Resume a paused study session.
    """
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    now = datetime.now().isoformat()
    
    # End the current pause period if there is one
    if study_sessions[session_id]["paused_times"] and study_sessions[session_id]["paused_times"][-1]["end"] is None:
        study_sessions[session_id]["paused_times"][-1]["end"] = now
    
    # Start a new active period
    study_sessions[session_id]["current_active_start"] = now
    
    return {"message": "Session resumed", "timestamp": now}

@app.post("/submit-client-metrics/{session_id}")
async def submit_client_metrics(session_id: str, metrics: ClientMetrics):
    """
    Receive client-side metrics including tracking mode information.
    This endpoint now handles all tracking data from the client.
    """
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Add metrics to session data
    study_sessions[session_id]["client_metrics"].append(metrics.dict())
    
    # Update server-side metrics with client-side data
    timestamp = metrics.timestamp
    
    # Store blink rate
    study_sessions[session_id]["metrics"]["blink_rate"].append({
        "timestamp": timestamp,
        "rate": metrics.blink_metrics.blink_rate
    })
    
    # Store tracking mode changes
    if metrics.tracking_mode:
        study_sessions[session_id]["metrics"]["tracking_modes"].append({
            "timestamp": timestamp,
            "mode": metrics.tracking_mode
        })
    
    # Check for concerning metrics and generate suggestions
    suggestions = []
    
    # Check blink rate
    if metrics.blink_metrics.blink_rate < 10:  # Less than 10 blinks per minute
        suggestions.append("Your blink rate is low. Consider taking a short break to rest your eyes.")
    
    # Check tracking mode
    if metrics.tracking_mode == "motion" or metrics.tracking_mode == "memory":
        suggestions.append("Face tracking intermittent. Consider adjusting your position or camera angle.")
    elif metrics.tracking_mode == "inactive":
        suggestions.append("No activity detected. Are you still studying?")
    
    # Check attention estimate
    if metrics.client_estimated_attention is not None and metrics.client_estimated_attention < 0.6:
        suggestions.append("You may be losing focus. Consider switching to a different topic or take a short break.")
    
    # Check for long periods without movement
    if metrics.seconds_since_last_movement and metrics.seconds_since_last_movement > 300:  # 5 minutes
        suggestions.append("You haven't moved in a while. Consider stretching or changing your position.")
    
    # If we have suggestions, randomly select one to avoid bombarding the user
    if suggestions:
        import random
        suggestion = random.choice(suggestions)
        
        # Add to distraction events if this indicates a distraction
        if "losing focus" in suggestion or "No activity" in suggestion:
            study_sessions[session_id]["metrics"]["distraction_events"].append({
                "timestamp": timestamp,
                "reason": suggestion
            })
        
        return {
            "message": "Client metrics received",
            "suggestion": suggestion
        }
    
    return {
        "message": "Client metrics received"
    }

#utils: generate_suggestions
def generate_suggestions(session_data, tracking_stats=None):
    """Generate personalized suggestions based on the session data and tracking statistics."""
    suggestions = []
    
    # Check for distractions
    distraction_count = len(session_data["metrics"]["distraction_events"])
    
    if distraction_count > 10:
        suggestions.append("You seemed to get distracted frequently. Consider finding a quieter study environment.")
    
    # Check blink rate from client metrics
    low_blink_rate = False
    if session_data["client_metrics"]:
        blink_rates = [
            metrics["blink_metrics"]["blink_rate"] 
            for metrics in session_data["client_metrics"]
            if "blink_metrics" in metrics and "blink_rate" in metrics["blink_metrics"]
        ]
        avg_blink_rate = sum(blink_rates) / len(blink_rates) if blink_rates else None
        
        if avg_blink_rate and avg_blink_rate < 10:  # Less than 10 blinks per minute
            low_blink_rate = True
            suggestions.append("Your blink rate was low, which may indicate eye strain. Try the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds.")
        elif avg_blink_rate and avg_blink_rate > 30:  # More than 30 blinks per minute
            suggestions.append("Your blink rate was high, which may indicate fatigue. Consider scheduling your study sessions when you're more alert.")
    
    # Check tracking stats for camera angle or positioning issues
    if tracking_stats:
        face_tracking_percent = tracking_stats.get('face', 0)
        
        if face_tracking_percent < 60:  # Less than 60% of time with direct face tracking
            suggestions.append("Your face was only detected directly for about {:.0f}% of your session. Consider adjusting your camera angle or position for better tracking.")
        
        if 'inactive' in tracking_stats and tracking_stats['inactive'] > 20:
            suggestions.append("There were significant periods with no activity detected. If you're taking breaks, consider using the pause feature.")
    
    # Add gaze-based suggestions
    has_gaze_metrics = False
    content_attention_percentage = 0
    
    # Check if we have gaze metrics
    for metrics in session_data["client_metrics"]:
        if "gaze_metrics" in metrics and metrics["gaze_metrics"] is not None:
            has_gaze_metrics = True
            if "looking_at_content_percentage" in metrics["gaze_metrics"]:
                content_attention_percentage += metrics["gaze_metrics"]["looking_at_content_percentage"]
    
    if has_gaze_metrics:
        # Calculate average content attention
        avg_content_attention = content_attention_percentage / len(session_data["client_metrics"])
        
        if avg_content_attention < 70:
            suggestions.append(f"You were looking at the study content only {avg_content_attention:.0f}% of the time. Try to minimize distractions in your environment.")
        elif avg_content_attention > 95:
            suggestions.append("Your visual focus is excellent! You maintained attention on the study content consistently.")

    # Check posture changes
    posture_change_count = len(session_data["metrics"]["posture_changes"])
    if posture_change_count > 10:
        suggestions.append("You changed posture frequently. Consider checking your chair ergonomics or taking shorter, more frequent breaks.")
    
    # Check session duration and effectiveness
    if "end_time" in session_data and session_data["end_time"] and "start_time" in session_data:
        start_time = datetime.fromisoformat(session_data["start_time"])
        end_time = datetime.fromisoformat(session_data["end_time"])
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        # Calculate effective study time vs total time if we have active times
        effective_minutes = duration_minutes
        if session_data.get("active_times"):
            active_seconds = sum(
                (datetime.fromisoformat(period["end"]) - datetime.fromisoformat(period["start"])).total_seconds()
                for period in session_data["active_times"]
            )
            effective_minutes = active_seconds / 60
        
        # Check if effective time is significantly less than total time
        if effective_minutes < duration_minutes * 0.7:
            suggestions.append(f"Your effective study time was only about {effective_minutes:.0f} minutes out of {duration_minutes:.0f} total minutes. Try to minimize interruptions for more efficient studying.")
        
        if duration_minutes > 90 and not low_blink_rate:
            suggestions.append("You had a long study session. Remember to take regular breaks to maintain focus and prevent eye strain.")
        
        # Check for very short sessions
        if duration_minutes < 15:
            suggestions.append("Your study session was quite short. Consider scheduling longer blocks of time for deeper focus and learning.")
    
    # Add general suggestions if we don't have many specific ones
    if len(suggestions) < 2:
        suggestions.append("For better focus, try the Pomodoro Technique: 25 minutes of focused study followed by a 5-minute break.")
        suggestions.append("Remember to hydrate regularly during study sessions to maintain optimal brain function.")
    
    return suggestions

@app.post("/end-session/{session_id}")
def end_session(session_id: str):
    """
    End a study session and calculate final metrics.
    """
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Ensure any active periods are properly closed
    if study_sessions[session_id].get("current_active_start"):
        now = datetime.now().isoformat()
        study_sessions[session_id]["active_times"].append({
            "start": study_sessions[session_id]["current_active_start"],
            "end": now
        })
    
    study_sessions[session_id]["end_time"] = datetime.now().isoformat()
    
    # Calculate session duration
    start_time = datetime.fromisoformat(study_sessions[session_id]["start_time"])
    end_time = datetime.fromisoformat(study_sessions[session_id]["end_time"])
    duration_minutes = (end_time - start_time).total_seconds() / 60
    
    # Calculate effective study time (excluding pauses)
    effective_duration_minutes = duration_minutes
    
    if study_sessions[session_id]["active_times"]:
        # Calculate based on active time periods
        active_time_seconds = 0
        for period in study_sessions[session_id]["active_times"]:
            period_start = datetime.fromisoformat(period["start"])
            period_end = datetime.fromisoformat(period["end"])
            active_time_seconds += (period_end - period_start).total_seconds()
        
        effective_duration_minutes = active_time_seconds / 60
    
    # Calculate metrics
    distraction_count = len(study_sessions[session_id]["metrics"]["distraction_events"])
    distraction_rate = distraction_count / max(1, effective_duration_minutes)
    
    # Calculate average focus time between distractions
    avg_focus_minutes = effective_duration_minutes / max(1, distraction_count + 1)
    
    # Analyze tracking modes
    tracking_modes = study_sessions[session_id]["metrics"]["tracking_modes"]
    tracking_stats = {}
    if tracking_modes:
        # Count occurrences of each tracking mode
        for entry in tracking_modes:
            mode = entry["mode"]
            if mode not in tracking_stats:
                tracking_stats[mode] = 0
            tracking_stats[mode] += 1
        
        # Convert to percentages
        total_entries = len(tracking_modes)
        for mode in tracking_stats:
            tracking_stats[mode] = (tracking_stats[mode] / total_entries) * 100
    
    # Get client-side blink metrics if available
    avg_blink_rate = None
    if study_sessions[session_id]["client_metrics"]:
        blink_rates = [
            metrics["blink_metrics"]["blink_rate"] 
            for metrics in study_sessions[session_id]["client_metrics"]
            if "blink_metrics" in metrics and "blink_rate" in metrics["blink_metrics"]
        ]
        avg_blink_rate = sum(blink_rates) / len(blink_rates) if blink_rates else None
    
    # Calculate posture changes
    posture_change_count = len(study_sessions[session_id]["metrics"]["posture_changes"])
    
    # Generate personalized suggestions
    suggestions = generate_suggestions(study_sessions[session_id], tracking_stats)
    
    summary = {
        "session_id": session_id,
        "duration_minutes": duration_minutes,
        "effective_study_minutes": effective_duration_minutes,
        "distraction_count": distraction_count,
        "distraction_rate_per_minute": distraction_rate,
        "average_focus_period_minutes": avg_focus_minutes,
        "posture_change_count": posture_change_count,
        "average_blink_rate": avg_blink_rate,
        "tracking_stats": tracking_stats,
        "suggestions": suggestions
    }
    
    return summary

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)