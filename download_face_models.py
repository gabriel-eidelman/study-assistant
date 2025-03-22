import os
import requests
import json
from pathlib import Path

def create_directory_if_not_exists(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def download_file(url, destination):
    """Download a file from a URL to a local destination with proper headers."""
    print(f"Downloading {url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        # For JSON files, validate the content
        if destination.suffix == '.json':
            try:
                # Try to parse the JSON to validate it
                json_content = response.json()
                
                # Write the prettified JSON to file
                with open(destination, 'w', encoding='utf-8') as f:
                    json.dump(json_content, f, indent=2)
                
                print(f"Successfully downloaded and validated JSON: {destination}")
                return True
            except json.JSONDecodeError as e:
                print(f"ERROR: Downloaded content is not valid JSON: {e}")
                return False
        else:
            # For binary files (weights)
            with open(destination, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded binary file: {destination}")
            return True
    else:
        print(f"Failed to download {url}, status code: {response.status_code}")
        return False

def main():
    """Download face-api.js model files from a reliable source."""
    # Create the static/models directory
    base_dir = Path("static/models")
    create_directory_if_not_exists(base_dir)
    
    # Alternative source for the face model files
    model_files = [
        # TinyFaceDetector model
        {
            'url': 'https://justadudewhohacks.github.io/face-api.js/models/tiny_face_detector_model-weights_manifest.json',
            'filename': 'tiny_face_detector_model-weights_manifest.json'
        },
        {
            'url': 'https://justadudewhohacks.github.io/face-api.js/models/tiny_face_detector_model-shard1',
            'filename': 'tiny_face_detector_model-shard1'
        },
        # FaceLandmark68 model
        {
            'url': 'https://justadudewhohacks.github.io/face-api.js/models/face_landmark_68_model-weights_manifest.json',
            'filename': 'face_landmark_68_model-weights_manifest.json'
        },
        {
            'url': 'https://justadudewhohacks.github.io/face-api.js/models/face_landmark_68_model-shard1',
            'filename': 'face_landmark_68_model-shard1'
        }
    ]
    
    # Download each file
    for file_info in model_files:
        url = file_info['url']
        filename = file_info['filename']
        destination = base_dir / filename
        success = download_file(url, destination)
        
        if not success:
            print(f"WARNING: Failed to download {filename}. Face detection may not work correctly.")
    
    print("\nModel download completed.")
    print("Please check the models in the static/models directory.")

if __name__ == "__main__":
    main()