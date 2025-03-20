import requests
import os
import sys
import time

def test_health_endpoint():
    """Test the health endpoint of the API."""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Health endpoint is working!")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ Health endpoint returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to health endpoint: {str(e)}")
        return False

def test_upload_endpoint(image_path):
    """Test the upload endpoint with a sample image."""
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post("http://localhost:8000/upload_image/", files=files)
        
        if response.status_code == 200:
            print("✅ Upload endpoint is working!")
            print("Response:")
            result = response.json()
            print(f"Extracted text: {result.get('extracted_text', 'None')}")
            print(f"Analysis: {result.get('analysis', 'None')}")
            return True
        else:
            print(f"❌ Upload endpoint returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to upload endpoint: {str(e)}")
        return False

def main():
    """Run the API tests."""
    print("Testing Medical Prescription Chatbot API...")
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(3)
    
    # Test health endpoint
    health_ok = test_health_endpoint()
    
    # Test upload endpoint if health is OK and image path is provided
    if health_ok and len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_upload_endpoint(image_path)
    elif health_ok:
        print("\n⚠️ No image path provided. To test the upload endpoint, run:")
        print("python test_api.py path/to/prescription_image.jpg")

if __name__ == "__main__":
    main() 