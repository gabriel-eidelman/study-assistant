import os
import cv2
import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import FaceAttributeType
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO
import requests


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Study Assistant API", description="Track facial features to optimize study sessions")

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

models_dir = os.path.join(static_dir, "models")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Mount static files directory
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Azure Face API credentials
face_key = os.getenv("AZURE_FACE_KEY")
face_endpoint = os.getenv("AZURE_FACE_ENDPOINT")

# Initialize Azure Face client
face_client = None
if face_key and face_endpoint:
    try:
        face_client = FaceClient(face_endpoint, CognitiveServicesCredentials(face_key))
        print(f"Face client initialized with endpoint: {face_endpoint}")
    except Exception as e:
        print(f"Error initializing Face client: {str(e)}")
else:
    print("Azure Face API credentials not found in environment variables")

# In-memory storage for study sessions, temp until using database
study_sessions = {}

# Pydantic models for data validation
class BlinkMetrics(BaseModel):
    blink_count: int
    blink_rate: float  # Blinks per minute
    avg_blink_duration: Optional[float] = None  # In milliseconds

class ClientMetrics(BaseModel):
    timestamp: str
    blink_metrics: BlinkMetrics
    seconds_since_last_movement: Optional[int] = None
    client_estimated_attention: Optional[float] = None  # 0-1 scale

@app.get("/")
def read_root():
    return {"message": "Study Assistant API is running"}

@app.get("/client", response_class=HTMLResponse)
async def get_client():
    """
    Serve the test client HTML page directly from FastAPI.
    """
    with open("test_client.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/detect-face/")
async def detect_face(file: UploadFile = File(...)):
    """
    Detect facial features using Azure Face API with minimal parameters.
    This endpoint only does basic detection without using restricted features.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read the image file
    contents = await file.read()
    
    try:
        # Detect faces with minimal parameters
        image_stream = BytesIO(contents)
        image_stream.seek(0)
        
        # Use minimal parameters to avoid Limited Access requirements
        detected_faces = face_client.face.detect_with_stream(
            image=image_stream,
            return_face_id=False,
            return_face_landmarks=True,  # Only get landmarks
            return_face_attributes=[],   # No attributes
            detection_model='detection_01'
        )
        
        if not detected_faces:
            return {"message": "No faces detected"}
        
        # Process the results - only use facial landmarks
        results = []
        for face in detected_faces:
            # Extract landmarks for our own analysis
            landmarks = face.face_landmarks
            
            # Detect head orientation using nose and eye positions
            nose_tip = [landmarks.nose_tip.x, landmarks.nose_tip.y]
            left_eye = [(landmarks.eye_left_inner.x + landmarks.eye_left_outer.x) / 2, 
                        (landmarks.eye_left_inner.y + landmarks.eye_left_outer.y) / 2]
            right_eye = [(landmarks.eye_right_inner.x + landmarks.eye_right_outer.x) / 2, 
                         (landmarks.eye_right_inner.y + landmarks.eye_right_outer.y) / 2]
                         
            # Calculate head orientation from landmarks
            head_orientation = estimate_head_orientation(left_eye, right_eye, nose_tip)
            
            # Calculate face size based on available landmarks
            # Use the distance between eyes and mouth for height approximation
            face_width = abs(landmarks.eye_left_outer.x - landmarks.eye_right_outer.x)
            
            # For height, we'll use the distance from eye level to mouth level
            # Since mouth_bottom isn't available, we'll use the average of mouth_left and mouth_right
            mouth_y = (landmarks.mouth_left.y + landmarks.mouth_right.y) / 2
            eye_y = (left_eye[1] + right_eye[1]) / 2
            face_height = abs(eye_y - mouth_y) * 1.5  # Multiply by 1.5 to account for forehead
            
            face_size = face_width * face_height
            
            # Estimate posture based on face position in frame
            posture_estimate = estimate_posture(face_size, nose_tip[1])
            
            face_data = {
                "landmarks": {
                    "eye_left_outer": [landmarks.eye_left_outer.x, landmarks.eye_left_outer.y],
                    "eye_left_top": [landmarks.eye_left_top.x, landmarks.eye_left_top.y],
                    "eye_left_bottom": [landmarks.eye_left_bottom.x, landmarks.eye_left_bottom.y],
                    "eye_left_inner": [landmarks.eye_left_inner.x, landmarks.eye_left_inner.y],
                    "eye_right_outer": [landmarks.eye_right_outer.x, landmarks.eye_right_outer.y],
                    "eye_right_top": [landmarks.eye_right_top.x, landmarks.eye_right_top.y],
                    "eye_right_bottom": [landmarks.eye_right_bottom.x, landmarks.eye_right_bottom.y],
                    "eye_right_inner": [landmarks.eye_right_inner.x, landmarks.eye_right_inner.y],
                    "nose_tip": [landmarks.nose_tip.x, landmarks.nose_tip.y],
                    "mouth_left": [landmarks.mouth_left.x, landmarks.mouth_left.y],
                    "mouth_right": [landmarks.mouth_right.x, landmarks.mouth_right.y]
                },
                "head_orientation": head_orientation,
                "posture": posture_estimate,
                "face_size": face_size
            }
            results.append(face_data)
        
        return {"faces": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

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
            "focus_periods": []
        },
        "end_time": None
    }
    return {"session_id": session_id, "message": "Study session started"}

@app.post("/simple-face-test/")
async def simple_face_test(file: UploadFile = File(...)):
    """
    A simplified endpoint to test face detection functionality with minimal features.
    Avoids all restricted features that require Limited Access approval.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image, got {file.content_type}")
    
    # Read the image file
    contents = await file.read()
    image_size = len(contents)
    
    # Diagnostic info
    diagnostic = {
        "file_size": image_size,
        "content_type": file.content_type,
        "filename": file.filename,
        "api_endpoint": face_endpoint,
        "api_key_available": face_key is not None,
        "sdk_version": "0.6.1"
    }
    
    if image_size == 0:
        return {
            "success": False,
            "error": "Empty image file",
            "diagnostic": diagnostic,
            "message": "Failed to detect faces - received empty file"
        }
    
    try:
        # Convert bytes to a file-like object
        image_stream = BytesIO(contents)
        image_stream.seek(0)  # Ensure we're at the start of the stream
        
        try:
            # Use absolute minimum parameters - just detect if faces exist
            print("Attempting basic face detection with minimal parameters...")
            
            detected_faces = face_client.face.detect_with_stream(
                image=image_stream,
                return_face_id=False,  # Don't request face IDs
                return_face_landmarks=False,  # Don't request landmarks
                return_face_attributes=[],  # No attributes
                detection_model='detection_01'  # Basic detection model
            )
            
            print(f"Detection successful! Found {len(detected_faces)} faces.")
            
            return {
                "success": True,
                "detected_faces": len(detected_faces),
                "diagnostic": diagnostic,
                "message": f"Successfully detected {len(detected_faces)} faces with minimal mode"
            }
            
        except Exception as face_error:
            error_str = str(face_error)
            print(f"Face detection error: {error_str}")
            
            return {
                "success": False,
                "error": error_str,
                "diagnostic": diagnostic,
                "message": "Failed to detect faces - even with minimal parameters"
            }
            
    except Exception as e:
        print(f"General processing error: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "diagnostic": diagnostic,
            "message": "Failed to process image"
        }

@app.post("/direct-face-test/")
async def direct_face_test(file: UploadFile = File(...)):
    """
    Test face detection using direct HTTP requests to Azure API with minimal features.
    Uses only the basic detection features that don't require Limited Access approval.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image, got {file.content_type}")
    
    # Read the image file
    contents = await file.read()
    image_size = len(contents)
    
    # Diagnostic info
    diagnostic = {
        "file_size": image_size,
        "content_type": file.content_type,
        "filename": file.filename,
        "api_endpoint": face_endpoint,
        "api_key_available": face_key is not None
    }
    
    if not face_key or not face_endpoint:
        return {
            "success": False,
            "error": "Azure Face API credentials not configured",
            "diagnostic": diagnostic
        }
    
    if image_size == 0:
        return {
            "success": False,
            "error": "Empty image file",
            "diagnostic": diagnostic
        }
    
    try:
        # Construct the direct API URL
        api_url = f"{face_endpoint}/face/v1.0/detect"
        
        # Set headers
        headers = {
            'Content-Type': file.content_type,
            'Ocp-Apim-Subscription-Key': face_key
        }
        
        # Use absolute minimum parameters to avoid requiring Limited Access
        params = {
            'returnFaceId': 'false',  # Don't get face IDs
            'returnFaceLandmarks': 'false',  # Don't get landmarks
            'returnFaceAttributes': '',  # No attributes
            'detectionModel': 'detection_01'  # Basic detection model
        }
        
        # Make direct API request
        response = requests.post(api_url, params=params, headers=headers, data=contents)
        
        # Check status code
        if response.status_code == 200:
            # Success
            faces = response.json()
            return {
                "success": True,
                "detected_faces": len(faces),
                "raw_response": faces,
                "diagnostic": diagnostic,
                "message": f"Successfully detected {len(faces)} faces using direct API call with minimal parameters"
            }
        else:
            # Error
            return {
                "success": False,
                "error": f"API returned status code {response.status_code}",
                "raw_response": response.text,
                "diagnostic": diagnostic,
                "message": "Failed to detect faces - direct API call failed"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "diagnostic": diagnostic,
            "message": "Failed to process image with direct API call"
        }

# Add this endpoint to serve the test page
@app.get("/test-page")
async def get_test_page():
    """
    Serve the API test page HTML file.
    """
    # If you put the file in a 'static' folder:
    return FileResponse("static/api-test-page.html")

@app.get("/check-face-api/")
def check_face_api():
    """
    Check if the Azure Face API credentials are working correctly.
    """
    try:
        # Print API details for debugging
        print(f"Face API Endpoint: {face_endpoint}")
        print(f"Face API Key available: {'Yes' if face_key else 'No'}")
        
        # Just return the API configuration information
        return {
            "status": "info",
            "message": "Azure Face API configuration",
            "endpoint": face_endpoint,
            "key_available": face_key is not None,
            "client_initialized": face_client is not None
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking API configuration: {str(e)}"
        }
    
@app.get("/check-azure-credentials/")
def check_azure_credentials():
    """
    Check if Azure Face API credentials are valid without making an API call.
    """
    # Print credentials for debugging (mask the key for security)
    key_status = "Not set" if not face_key else f"Set ({face_key[:3]}...{face_key[-3:] if len(face_key) > 6 else ''})"
    endpoint_status = "Not set" if not face_endpoint else face_endpoint
    
    # Basic validation of the credentials format
    valid_key_format = face_key and len(face_key) > 10
    valid_endpoint_format = face_endpoint and face_endpoint.startswith("https://") and "cognitiveservices.azure.com" in face_endpoint
    
    if not valid_key_format or not valid_endpoint_format:
        return {
            "valid": False,
            "message": "Azure Face API credentials are not properly formatted",
            "key_status": key_status,
            "endpoint_status": endpoint_status,
            "key_format_valid": valid_key_format,
            "endpoint_format_valid": valid_endpoint_format
        }
    
    # If the credentials look valid in format, return success
    # This doesn't guarantee they work, but it's a start
    return {
        "valid": True,
        "message": "Azure Face API credentials are properly formatted",
        "key_status": key_status,
        "endpoint_status": endpoint_status,
        "note": "This check only verifies credential format, not API access. Try the 'Test Face Detection' to confirm API functionality."
    }

@app.post("/process-frame/{session_id}")
async def process_frame(session_id: str, file: UploadFile = File(...)):
    """
    Process a video frame from the study session using facial landmarks.
    Uses only basic detection features that don't require Limited Access approval.
    
    This endpoint now focuses more on head position and posture, while 
    relying on client-side metrics for blink detection.
    """
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read the image file
    contents = await file.read()
    
    # Log file size for debugging
    print(f"Image size: {len(contents)} bytes")
    
    try:
        # Convert bytes to a file-like object and reset position
        image_stream = BytesIO(contents)
        image_stream.seek(0)  # Ensure we're at the start of the stream
        
        # Try to perform a minimal face detection
        try:
            detected_faces = face_client.face.detect_with_stream(
                image=image_stream,
                return_face_id=False,
                return_face_landmarks=True,  # Only get landmarks
                return_face_attributes=[],   # No attributes
                detection_model='detection_01'
            )
            
            timestamp = datetime.now().isoformat()
            
            if not detected_faces:
                # When no faces are detected, assume the user was distracted and log the event
                study_sessions[session_id]["metrics"]["distraction_events"].append({
                    "timestamp": timestamp,
                    "reason": "no_face_detected"
                })
                return {"message": "No face detected", "suggestion": "Are you still there?"}
            
            # For simplicity, just process the first face detected
            face = detected_faces[0]
            landmarks = face.face_landmarks
            
            # Calculate head orientation using facial landmarks
            nose_tip = [landmarks.nose_tip.x, landmarks.nose_tip.y]
            left_eye = [(landmarks.eye_left_inner.x + landmarks.eye_left_outer.x) / 2, 
                        (landmarks.eye_left_inner.y + landmarks.eye_left_outer.y) / 2]
            right_eye = [(landmarks.eye_right_inner.x + landmarks.eye_right_outer.x) / 2, 
                         (landmarks.eye_right_inner.y + landmarks.eye_right_outer.y) / 2]
                         
            # Calculate head orientation from landmarks
            head_orientation = estimate_head_orientation(left_eye, right_eye, nose_tip)
            
            # Calculate face size for posture estimation
            face_width = abs(landmarks.eye_left_outer.x - landmarks.eye_right_outer.x)
            
            # For height, use the distance from eye level to mouth level
            mouth_y = (landmarks.mouth_left.y + landmarks.mouth_right.y) / 2
            eye_y = (left_eye[1] + right_eye[1]) / 2
            face_height = abs(eye_y - mouth_y) * 1.5  # Multiply by 1.5 to account for forehead
            
            face_size = face_width * face_height
            
            # Estimate posture based on face position in frame
            posture_estimate = estimate_posture(face_size, nose_tip[1])
            
            # Store frame data
            frame_data = {
                "timestamp": timestamp,
                "head_orientation": head_orientation,
                "posture": posture_estimate,
                "face_size": face_size
            }
            
            # Add frame data to session
            study_sessions[session_id]["frames"].append(frame_data)
            
            # Head pose-based distraction detection using our estimated yaw
            head_yaw = abs(head_orientation["yaw"])
            if head_yaw > 15:  # If user was looking to the side significantly, assume distraction event
                study_sessions[session_id]["metrics"]["distraction_events"].append({
                    "timestamp": timestamp,
                    "reason": "looking_away",
                    "yaw": head_yaw
                })
                return {
                    "message": "Possible distraction detected",
                    "suggestion": "Try to focus on your study materials",
                    "details": frame_data
                }
            
            # Check posture
            if posture_estimate["is_slouching"]:
                return {
                    "message": "Posture check",
                    "suggestion": "You might be slouching. Try to sit up straight for better focus.",
                    "details": frame_data
                }
                
            # If we have enough frames, check for significant posture changes
            # which could indicate restlessness or discomfort
            if len(study_sessions[session_id]["frames"]) > 5:
                recent_frames = study_sessions[session_id]["frames"][-5:]
                face_sizes = [frame["face_size"] for frame in recent_frames]
                size_variance = np.var(face_sizes) if len(face_sizes) > 1 else 0
                
                # High variance in face size indicates movement toward/away from camera
                if size_variance > 1000:  # Threshold determined empirically
                    study_sessions[session_id]["metrics"]["posture_changes"].append({
                        "timestamp": timestamp,
                        "variance": size_variance
                    })
                    
                    if len(study_sessions[session_id]["metrics"]["posture_changes"]) > 5:
                        return {
                            "message": "Frequent movement detected",
                            "suggestion": "You seem restless. Consider taking a short break.",
                            "details": frame_data
                        }
            
            return {
                "message": "Frame processed",
                "details": frame_data
            }
            
        except Exception as face_error:
            print(f"Detailed face detection error: {str(face_error)}")
            return {
                "message": "Error during face detection",
                "error": str(face_error)
            }
            
    except Exception as e:
        print(f"General frame processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")

@app.post("/submit-client-metrics/{session_id}")
async def submit_client_metrics(session_id: str, metrics: ClientMetrics):
    """
    New endpoint to receive client-side metrics, including blink detection
    that's performed at a higher frequency in the browser.
    """
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Add client metrics to session data
    study_sessions[session_id]["client_metrics"].append(metrics.dict())
    
    # Update server-side metrics with client-side data
    study_sessions[session_id]["metrics"]["blink_rate"].append({
        "timestamp": metrics.timestamp,
        "rate": metrics.blink_metrics.blink_rate
    })
    
    # Check for concerning blink rate
    if metrics.blink_metrics.blink_rate < 10:  # Less than 10 blinks per minute
        return {
            "message": "Low blink rate detected",
            "suggestion": "Your blink rate is low. Consider taking a short break to rest your eyes.",
            "client_metrics_received": True
        }
    
    # Check for client-estimated attention
    if metrics.client_estimated_attention is not None and metrics.client_estimated_attention < 0.6:
        return {
            "message": "Attention level decreasing",
            "suggestion": "You may be losing focus. Consider switching to a different topic or take a short break.",
            "client_metrics_received": True
        }
    
    return {
        "message": "Client metrics received",
        "client_metrics_received": True
    }

@app.post("/end-session/{session_id}")
def end_session(session_id: str):
    """
    End a study session and calculate final metrics.
    """
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    study_sessions[session_id]["end_time"] = datetime.now().isoformat()
    
    # Calculate session duration
    start_time = datetime.fromisoformat(study_sessions[session_id]["start_time"])
    end_time = datetime.fromisoformat(study_sessions[session_id]["end_time"])
    duration_minutes = (end_time - start_time).total_seconds() / 60
    
    # Calculate metrics
    distraction_count = len(study_sessions[session_id]["metrics"]["distraction_events"])
    distraction_rate = distraction_count / max(1, duration_minutes)
    
    # Calculate average focus time between distractions
    avg_focus_minutes = duration_minutes / max(1, distraction_count + 1)
    
    # Get client-side blink metrics if available
    avg_blink_rate = None
    if study_sessions[session_id]["client_metrics"]:
        blink_rates = [
            metrics["blink_metrics"]["blink_rate"] 
            for metrics in study_sessions[session_id]["client_metrics"]
        ]
        avg_blink_rate = sum(blink_rates) / len(blink_rates) if blink_rates else None
    
    # Calculate posture changes
    posture_change_count = len(study_sessions[session_id]["metrics"]["posture_changes"])
    
    summary = {
        "session_id": session_id,
        "duration_minutes": duration_minutes,
        "distraction_count": distraction_count,
        "distraction_rate_per_minute": distraction_rate,
        "average_focus_period_minutes": avg_focus_minutes,
        "posture_change_count": posture_change_count,
        "average_blink_rate": avg_blink_rate,
        "suggestions": generate_suggestions(study_sessions[session_id])
    }
    
    return summary

@app.get("/session/{session_id}")
def get_session(session_id: str):
    """
    Get details of a specific study session.
    """
    if session_id not in study_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return study_sessions[session_id]

# Utility functions

def estimate_head_orientation(left_eye, right_eye, nose_tip):
    """
    Roughly estimate head orientation using only facial landmarks
    when no head pose attributes are available.
    """
    # Calculate eye midpoint
    eye_midpoint = [(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2]
    
    # Distance between eyes
    eye_distance = ((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)**0.5
    
    # Calculate yaw (horizontal orientation)
    # If nose tip is significantly to the left or right of the eye midpoint
    # relative to the eye distance, the person is looking to the side
    horizontal_diff = nose_tip[0] - eye_midpoint[0]
    normalized_yaw = horizontal_diff / (eye_distance * 0.5)  # Normalize by half eye distance
    estimated_yaw = normalized_yaw * 30  # Scale to roughly match degrees
    
    # Calculate pitch (vertical orientation)
    # If nose tip is above or below eye midpoint
    vertical_diff = nose_tip[1] - eye_midpoint[1]
    normalized_pitch = vertical_diff / eye_distance
    estimated_pitch = normalized_pitch * 25  # Scale to roughly match degrees
    
    return {
        "yaw": estimated_yaw,       # Negative: looking left, Positive: looking right
        "pitch": estimated_pitch,   # Negative: looking up, Positive: looking down
        "is_looking_away": abs(estimated_yaw) > 15
    }

def estimate_posture(face_size, nose_y_position):
    """
    Estimate if the user is slouching or has poor posture based on face size
    and vertical position in the frame.
    
    A smaller face size may indicate the user is leaning back.
    A lower nose position may indicate slouching.
    """
    # These thresholds would need to be calibrated for your specific application
    return {
        "face_size": face_size,
        "nose_y_position": nose_y_position,
        "is_slouching": nose_y_position > 0.6,  # If nose is in lower part of frame
        "is_leaning_back": face_size < 5000     # Threshold needs calibration
    }

def generate_suggestions(session_data):
    """Generate personalized suggestions based on the session data."""
    suggestions = []
    
    # Example suggestions based on simple heuristics
    distraction_count = len(session_data["metrics"]["distraction_events"])
    
    if distraction_count > 10:
        suggestions.append("You seemed to get distracted frequently. Consider finding a quieter study environment.")
    
    # Check blink rate from client metrics
    low_blink_rate = False
    if session_data["client_metrics"]:
        blink_rates = [
            metrics["blink_metrics"]["blink_rate"] 
            for metrics in session_data["client_metrics"]
        ]
        avg_blink_rate = sum(blink_rates) / len(blink_rates) if blink_rates else None
        
        if avg_blink_rate and avg_blink_rate < 10:  # Less than 10 blinks per minute
            low_blink_rate = True
            suggestions.append("Your blink rate was low, which may indicate eye strain. Try the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds.")
    
    # Check posture changes
    posture_change_count = len(session_data["metrics"]["posture_changes"])
    if posture_change_count > 10:
        suggestions.append("You changed posture frequently. Consider checking your chair ergonomics or taking shorter, more frequent breaks.")
    
    # Check session duration for general advice
    if "end_time" in session_data and session_data["end_time"] and "start_time" in session_data:
        start_time = datetime.fromisoformat(session_data["start_time"])
        end_time = datetime.fromisoformat(session_data["end_time"])
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        if duration_minutes > 90 and not low_blink_rate:
            suggestions.append("You had a long study session. Remember to take regular breaks to maintain focus and prevent eye strain.")
    
    return suggestions

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)