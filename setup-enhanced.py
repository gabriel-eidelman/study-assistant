import os
import shutil
import subprocess
import sys

def create_directory_if_not_exists(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        # Read requirements.txt
        with open("requirements.txt", "r") as f:
            requirements = f.read().splitlines()
        
        # Remove Azure Face API dependency since we're not using it anymore
        requirements = [req for req in requirements if "azure-cognitiveservices-vision-face" not in req]
        
        # Create updated requirements file
        with open("requirements_enhanced.txt", "w") as f:
            f.write("\n".join(requirements))
        
        # Install requirements
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_enhanced.txt"])
        
        print("Dependencies installed successfully.")
        return True
    except Exception as e:
        print(f"Error installing dependencies: {str(e)}")
        return False

def copy_face_api_models():
    """
    Copy the face-api.js model files to the static directory.
    These are needed for client-side face detection.
    """
    # Create the static/models directory
    models_dir = os.path.join("static", "models")
    create_directory_if_not_exists(models_dir)
    
    # Check if we have the model files
    model_files = ["face_landmark_68_model-weights_manifest.json", "face_landmark_68_model-shard1",
                  "tiny_face_detector_model-weights_manifest.json", "tiny_face_detector_model-shard1"]
    
    # Count how many files we have in the current directory
    found_files = [f for f in model_files if os.path.exists(f)]
    
    if found_files:
        print(f"Found {len(found_files)} face-api.js model files in current directory")
        
        # Copy files to static/models
        for file in found_files:
            dest_path = os.path.join(models_dir, file)
            print(f"Copying {file} to {dest_path}")
            shutil.copy(file, dest_path)
    else:
        print("No face model files found in current directory.")
        print("You'll need to run download_face_models.py to download them.")
        print("Client-side face detection won't work without these files.")

def setup_enhanced_client():
    """
    Set up the enhanced client HTML file.
    """
    # Create static directory
    create_directory_if_not_exists("static")
    
    # Check if our enhanced client file is in this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    client_source = os.path.join(script_dir, "enhanced_client.html")
    
    if os.path.exists("enhanced-study-client.html"):
        client_source = "enhanced-study-client.html"
    
    if os.path.exists(client_source):
        print(f"Found enhanced client at {client_source}")
        shutil.copy(client_source, "enhanced_client.html")
        print("Copied enhanced client to enhanced_client.html")
    else:
        print("Enhanced client HTML file not found.")
        print("You'll need to create enhanced_client.html manually.")

def setup_enhanced_app():
    """
    Set up the enhanced app.py file.
    """
    if os.path.exists("enhanced-app.py"):
        print("Found enhanced app at enhanced-app.py")
        
        # Backup original app.py
        if os.path.exists("app.py"):
            backup_path = "app.py.backup"
            if os.path.exists(backup_path):
                os.remove(backup_path)  # Remove existing backup
            shutil.copy("app.py", backup_path)
            print("Created backup of original app.py at app.py.backup")
        
        # Copy enhanced app
        shutil.copy("enhanced-app.py", "app.py")
        print("Copied enhanced app to app.py")
    else:
        print("Enhanced app.py file not found.")
        print("You'll need to modify app.py manually.")

def main():
    """Run the complete setup process."""
    print("==================================================")
    print("      Enhanced Study Assistant Setup              ")
    print("==================================================")
    
    # Create needed directories
    create_directory_if_not_exists("static")
    
    # Check and install dependencies
    if not check_dependencies():
        print("Error with dependencies. Setup incomplete.")
        return
    
    # Copy face-api.js model files
    copy_face_api_models()
    
    # Set up enhanced client
    setup_enhanced_client()
    
    # Set up enhanced app
    setup_enhanced_app()
    
    print("\n==================================================")
    print("Setup complete! Follow these steps to run the app:")
    print("==================================================")
    print("1. Start the application:")
    print("   python app.py")
    print("2. Visit http://localhost:8000/client in your browser")
    print("\nKey improvements in the enhanced version:")
    print("- Face tracking is now handled entirely on the client side")
    print("- Tracking works even when face is partially visible or off-camera")
    print("- Motion detection as a fallback when face isn't visible")
    print("- Improved analytics and suggestions")
    print("==================================================")

if __name__ == "__main__":
    main()