from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import easyocr
import logging
from typing import Dict, Any, List, Optional
import os
import json
import base64
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="Medical Prescription Chatbot")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR
reader = easyocr.Reader(['en'])

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess the image for better OCR results."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better text extraction
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary)
    
    return denoised

def crop_image(image: np.ndarray, crop_data: Dict) -> np.ndarray:
    """Crop the image based on the provided coordinates."""
    try:
        if crop_data and all(key in crop_data for key in ['x', 'y', 'width', 'height']):
            x = int(crop_data['x'])
            y = int(crop_data['y'])
            width = int(crop_data['width'])
            height = int(crop_data['height'])
            
            # Ensure coordinates are within image boundaries
            height_img, width_img = image.shape[:2]
            x = max(0, min(x, width_img - 1))
            y = max(0, min(y, height_img - 1))
            width = max(1, min(width, width_img - x))
            height = max(1, min(height, height_img - y))
            
            # Crop the image
            cropped_image = image[y:y+height, x:x+width]
            return cropped_image
        return image
    except Exception as e:
        logger.error(f"Error cropping image: {str(e)}")
        return image

def extract_text_from_image(image: np.ndarray) -> str:
    """Extract text from the image using EasyOCR."""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Perform OCR with confidence threshold
        result = reader.readtext(processed_image, min_size=10, width_ths=0.7, height_ths=0.7)
        
        # Extract text from results, filtering by confidence
        extracted_text = ""
        for detection in result:
            if detection[2] > 0.5:  # confidence threshold
                extracted_text += detection[1] + "\n"
        
        return extracted_text.strip()
    except Exception as e:
        logger.error(f"Error in OCR processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing image")

def analyze_prescription(text: str) -> Dict[str, Any]:
    """Analyze the prescription text using rule-based approach."""
    try:
        # Initialize analysis dictionary
        analysis = {
            "medications": [],
            "dosages": [],
            "frequencies": [],
            "durations": []
        }
        
        # Look for common prescription elements
        lines = text.split('\n')
        for line in lines:
            line = line.strip().lower()
            
            # Check for medication names (common endings)
            if any(ending in line for ending in ['ol', 'in', 'um', 'ide', 'one', 'cin', 'xin']):
                # Extract medication name (simple approach)
                words = line.split()
                for word in words:
                    if any(ending in word for ending in ['ol', 'in', 'um', 'ide', 'one', 'cin', 'xin']):
                        if len(word) > 3:  # Avoid short words
                            analysis["medications"].append(word)
            
            # Check for dosage
            if any(unit in line for unit in ['mg', 'ml', 'g', 'mcg']):
                # Extract dosage information
                for unit in ['mg', 'ml', 'g', 'mcg']:
                    if unit in line:
                        # Find numbers before unit
                        parts = line.split(unit)
                        if parts and parts[0]:
                            words = parts[0].split()
                            if words:
                                # Get the last word before the unit
                                dosage = words[-1] + unit
                                analysis["dosages"].append(dosage)
            
            # Check for frequency
            frequency_terms = ['daily', 'twice', 'times', 'hourly', 'weekly', 'every']
            if any(term in line for term in frequency_terms):
                for term in frequency_terms:
                    if term in line:
                        # Extract the phrase containing the frequency term
                        index = line.find(term)
                        start = max(0, index - 10)
                        end = min(len(line), index + 20)
                        frequency = line[start:end].strip()
                        analysis["frequencies"].append(frequency)
                        break
            
            # Check for duration
            duration_terms = ['days', 'weeks', 'months', 'for']
            if any(term in line for term in duration_terms):
                for term in duration_terms:
                    if term in line:
                        # Extract the phrase containing the duration term
                        index = line.find(term)
                        start = max(0, index - 10)
                        end = min(len(line), index + 10)
                        duration = line[start:end].strip()
                        analysis["durations"].append(duration)
                        break
        
        # Remove duplicates
        for key in analysis:
            analysis[key] = list(set(analysis[key]))
        
        # If no elements were found, add a message
        if not any(analysis.values()):
            analysis["message"] = "No specific prescription elements identified."
        
        return analysis
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return {"error": "Could not analyze the prescription text"}

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the home page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical Prescription Analyzer</title>
        <!-- Google Fonts -->
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <!-- Material Icons -->
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <!-- Cropper.js CSS -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.css">
        <style>
            :root {
                --primary-color: #4361ee;
                --primary-light: #4895ef;
                --primary-dark: #3f37c9;
                --secondary-color: #4cc9f0;
                --accent-color: #f72585;
                --success-color: #4CAF50;
                --warning-color: #ff9800;
                --error-color: #f44336;
                --background-color: #f8f9fa;
                --card-color: #ffffff;
                --text-primary: #343a40;
                --text-secondary: #6c757d;
                --border-color: #dee2e6;
                --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.05);
                --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
                --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
                --radius-sm: 4px;
                --radius-md: 8px;
                --radius-lg: 12px;
                --transition: all 0.3s ease;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--background-color);
                color: var(--text-primary);
                line-height: 1.6;
                padding: 0;
                margin: 0;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 1rem;
            }
            
            /* Header styles */
            .app-header {
                background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
                color: white;
                padding: 2rem 0;
                text-align: center;
                box-shadow: var(--shadow-md);
                margin-bottom: 2rem;
            }
            
            .header-content {
                max-width: 800px;
                margin: 0 auto;
            }
            
            .app-title {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            
            .app-subtitle {
                font-size: 1.1rem;
                font-weight: 400;
                opacity: 0.9;
                max-width: 600px;
                margin: 0 auto;
            }
            
            /* Card styles */
            .card {
                background-color: var(--card-color);
                border-radius: var(--radius-md);
                box-shadow: var(--shadow-md);
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                transition: var(--transition);
            }
            
            .card:hover {
                box-shadow: var(--shadow-lg);
            }
            
            .card-title {
                color: var(--primary-color);
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .card-title .material-icons {
                font-size: 1.5rem;
            }
            
            /* Upload area styles */
            .upload-container {
                text-align: center;
                border: 2px dashed var(--primary-light);
                border-radius: var(--radius-md);
                padding: 2rem;
                margin-bottom: 1.5rem;
                background-color: rgba(67, 97, 238, 0.05);
                transition: var(--transition);
            }
            
            .upload-container:hover {
                background-color: rgba(67, 97, 238, 0.08);
                border-color: var(--primary-color);
            }
            
            .upload-icon {
                font-size: 3rem;
                color: var(--primary-color);
                margin-bottom: 1rem;
            }
            
            .upload-text {
                margin-bottom: 1.5rem;
                color: var(--text-secondary);
            }
            
            .file-input-container {
                position: relative;
                margin: 1rem auto;
                max-width: 300px;
            }
            
            .file-input {
                position: absolute;
                left: 0;
                top: 0;
                opacity: 0;
                width: 100%;
                height: 100%;
                cursor: pointer;
            }
            
            /* Button styles */
            .btn {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 0.5rem;
                background-color: var(--primary-color);
                color: white;
                border: none;
                border-radius: var(--radius-sm);
                padding: 0.75rem 1.5rem;
                font-size: 1rem;
                font-weight: 500;
                cursor: pointer;
                transition: var(--transition);
                box-shadow: var(--shadow-sm);
            }
            
            .btn:hover {
                background-color: var(--primary-dark);
                box-shadow: var(--shadow-md);
                transform: translateY(-1px);
            }
            
            .btn:active {
                transform: translateY(0);
                box-shadow: var(--shadow-sm);
            }
            
            .btn-secondary {
                background-color: white;
                color: var(--primary-color);
                border: 1px solid var(--primary-color);
            }
            
            .btn-secondary:hover {
                background-color: rgba(67, 97, 238, 0.05);
            }
            
            .btn-success {
                background-color: var(--success-color);
            }
            
            .btn-success:hover {
                background-color: #388E3C;
            }
            
            .btn-error {
                background-color: var(--error-color);
            }
            
            .btn-error:hover {
                background-color: #D32F2F;
            }
            
            /* File list styles */
            .file-list {
                margin-top: 1.5rem;
                max-height: 250px;
                overflow-y: auto;
                background-color: var(--card-color);
                border-radius: var(--radius-md);
                padding: 0.5rem;
                border: 1px solid var(--border-color);
                display: none;
            }
            
            .file-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.75rem 1rem;
                border-bottom: 1px solid var(--border-color);
                transition: var(--transition);
            }
            
            .file-item:last-child {
                border-bottom: none;
            }
            
            .file-item:hover {
                background-color: rgba(67, 97, 238, 0.05);
            }
            
            .file-name {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-weight: 500;
            }
            
            .file-name .material-icons {
                color: var(--text-secondary);
            }
            
            .file-badge {
                display: inline-block;
                background-color: rgba(67, 97, 238, 0.1);
                color: var(--primary-color);
                padding: 0.25rem 0.5rem;
                border-radius: 50px;
                font-size: 0.75rem;
                font-weight: 600;
                margin-left: 0.5rem;
            }
            
            .file-actions {
                display: flex;
                gap: 0.5rem;
            }
            
            .action-btn {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 2rem;
                height: 2rem;
                border-radius: 50%;
                background-color: transparent;
                border: none;
                cursor: pointer;
                transition: var(--transition);
            }
            
            .action-btn:hover {
                background-color: rgba(67, 97, 238, 0.1);
            }
            
            .action-btn.crop {
                color: var(--primary-color);
            }
            
            .action-btn.remove {
                color: var(--error-color);
            }
            
            .analyze-btn-container {
                display: flex;
                justify-content: center;
                margin-top: 1.5rem;
            }
            
            .analyze-btn {
                padding: 0.75rem 2rem;
                font-size: 1.1rem;
            }
            
            /* Loading spinner */
            .loading {
                display: none;
                text-align: center;
                margin: 2rem 0;
            }
            
            .spinner {
                width: 40px;
                height: 40px;
                margin: 0 auto;
                border: 4px solid rgba(0, 0, 0, 0.1);
                border-radius: 50%;
                border-left-color: var(--primary-color);
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .loading-text {
                margin-top: 1rem;
                color: var(--text-secondary);
                font-weight: 500;
            }
            
            /* Results styles */
            .results {
                display: none;
            }
            
            .result-item {
                margin-bottom: 2rem;
            }
            
            .result-header {
                background-color: var(--primary-color);
                color: white;
                padding: 1rem 1.5rem;
                border-radius: var(--radius-md) var(--radius-md) 0 0;
                font-weight: 600;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            
            .result-body {
                background-color: var(--card-color);
                border: 1px solid var(--border-color);
                border-top: none;
                border-radius: 0 0 var(--radius-md) var(--radius-md);
                overflow: hidden;
            }
            
            .result-tabs {
                display: flex;
                border-bottom: 1px solid var(--border-color);
            }
            
            .result-tab {
                padding: 0.75rem 1.5rem;
                cursor: pointer;
                font-weight: 500;
                transition: var(--transition);
                color: var(--text-secondary);
                border-bottom: 2px solid transparent;
            }
            
            .result-tab.active {
                color: var(--primary-color);
                border-bottom: 2px solid var(--primary-color);
            }
            
            .result-tab:hover:not(.active) {
                color: var(--text-primary);
                background-color: rgba(0, 0, 0, 0.02);
            }
            
            .result-content {
                padding: 1.5rem;
            }
            
            .result-section {
                display: none;
            }
            
            .result-section.active {
                display: block;
            }
            
            .section-title {
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 1rem;
                color: var(--primary-dark);
            }
            
            .extraction-text {
                font-family: 'Courier New', monospace;
                background-color: rgba(0, 0, 0, 0.03);
                padding: 1rem;
                border-radius: var(--radius-sm);
                white-space: pre-wrap;
                overflow-x: auto;
                font-size: 0.9rem;
                line-height: 1.5;
            }
            
            .analysis-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem;
                margin-top: 1rem;
            }
            
            .analysis-card {
                background-color: rgba(67, 97, 238, 0.05);
                border-radius: var(--radius-sm);
                padding: 1rem;
                border-left: 3px solid var(--primary-light);
            }
            
            .analysis-title {
                font-weight: 600;
                color: var(--primary-dark);
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .analysis-list {
                list-style-type: none;
            }
            
            .analysis-list li {
                padding: 0.5rem 0;
                border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            }
            
            .analysis-list li:last-child {
                border-bottom: none;
            }
            
            .empty-message {
                color: var(--text-secondary);
                font-style: italic;
            }
            
            /* Modal styles */
            .modal-overlay {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: rgba(0, 0, 0, 0.75);
                z-index: 1000;
                backdrop-filter: blur(5px);
                animation: fadeIn 0.3s ease;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            .modal-content {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background-color: var(--card-color);
                border-radius: var(--radius-lg);
                max-width: 90%;
                width: 800px;
                box-shadow: var(--shadow-lg);
                overflow: hidden;
                display: flex;
                flex-direction: column;
                max-height: 90vh;
            }
            
            .modal-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 1rem 1.5rem;
                background-color: var(--primary-color);
                color: white;
            }
            
            .modal-title {
                font-size: 1.2rem;
                font-weight: 600;
            }
            
            .modal-close {
                background: none;
                border: none;
                color: white;
                cursor: pointer;
                font-size: 1.5rem;
                display: flex;
                align-items: center;
                justify-content: center;
                width: 2rem;
                height: 2rem;
                border-radius: 50%;
                transition: var(--transition);
            }
            
            .modal-close:hover {
                background-color: rgba(255, 255, 255, 0.2);
            }
            
            .modal-body {
                padding: 1.5rem;
                overflow-y: auto;
                flex: 1;
            }
            
            .error-message {
                background-color: rgba(244, 67, 54, 0.1);
                color: var(--error-color);
                padding: 0.75rem 1rem;
                border-radius: var(--radius-sm);
                margin-bottom: 1rem;
                display: none;
            }
            
            .crop-container {
                background-color: #f0f0f0;
                text-align: center;
                position: relative;
                margin-bottom: 1.5rem;
                border-radius: var(--radius-sm);
                overflow: hidden;
            }
            
            .crop-container img {
                max-width: 100%;
                max-height: 500px;
                display: block;
                margin: 0 auto;
            }
            
            .modal-footer {
                display: flex;
                justify-content: flex-end;
                gap: 0.75rem;
                padding: 1rem 1.5rem;
                background-color: rgba(0, 0, 0, 0.02);
                border-top: 1px solid var(--border-color);
            }
            
            /* Responsive styles */
            @media (max-width: 768px) {
                .app-title {
                    font-size: 2rem;
                }
                
                .card {
                    padding: 1.25rem;
                }
                
                .upload-container {
                    padding: 1.5rem 1rem;
                }
                
                .analysis-grid {
                    grid-template-columns: 1fr;
                }
                
                .result-tabs {
                    flex-wrap: wrap;
                }
                
                .result-tab {
                    flex: 1;
                    text-align: center;
                    padding: 0.75rem 0.5rem;
                }
            }
            
            @media (max-width: 480px) {
                .container {
                    padding: 0.75rem;
                }
                
                .app-title {
                    font-size: 1.75rem;
                }
                
                .app-subtitle {
                    font-size: 1rem;
                }
                
                .file-item {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 0.5rem;
                }
                
                .file-actions {
                    align-self: flex-end;
                }
            }
        </style>
    </head>
    <body>
        <header class="app-header">
            <div class="header-content">
                <h1 class="app-title">Medical Prescription Analyzer</h1>
                <p class="app-subtitle">Upload clear images of medical prescriptions for automated analysis and text extraction</p>
            </div>
        </header>
        
        <div class="container">
            <div class="card">
                <h2 class="card-title">
                    <span class="material-icons">cloud_upload</span>Upload Prescriptions
                </h2>
                
                <div class="upload-container">
                    <div class="upload-icon">
                        <span class="material-icons">image</span>
                    </div>
                    <p class="upload-text">Drag and drop your prescription images or click to browse</p>
                    
                    <div class="file-input-container">
                        <button class="btn">
                            <span class="material-icons">add_photo_alternate</span>
                            Select Images
                        </button>
                        <input type="file" id="prescription-image" class="file-input" accept="image/*" multiple>
                    </div>
                </div>
                
                <div id="file-list" class="file-list"></div>
                
                <div class="analyze-btn-container">
                    <button id="upload-btn" class="btn analyze-btn">
                        <span class="material-icons">analytics</span>
                        Analyze Prescriptions
                    </button>
                </div>
            </div>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p class="loading-text">Processing prescriptions, please wait...</p>
            </div>
            
            <div id="results" class="results"></div>
        </div>
        
        <!-- Crop Modal -->
        <div id="crop-modal" class="modal-overlay">
            <div class="modal-content">
                <div class="modal-header">
                    <h3 class="modal-title">Crop Prescription Image</h3>
                    <button id="close-crop-btn" class="modal-close">
                        <span class="material-icons">close</span>
                    </button>
                </div>
                
                <div class="modal-body">
                    <div id="error-message" class="error-message"></div>
                    <div class="crop-container">
                        <img id="crop-image" src="" alt="Image to crop">
                    </div>
                </div>
                
                <div class="modal-footer">
                    <button id="cancel-crop-btn" class="btn btn-secondary">Cancel</button>
                    <button id="crop-btn" class="btn btn-success">
                        <span class="material-icons">check</span>
                        Apply Crop
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Cropper.js Script -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.js"></script>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // DOM elements
                const fileInput = document.getElementById('prescription-image');
                const fileList = document.getElementById('file-list');
                const uploadBtn = document.getElementById('upload-btn');
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');
                
                // Crop modal elements
                const cropModal = document.getElementById('crop-modal');
                const cropImage = document.getElementById('crop-image');
                const cropBtn = document.getElementById('crop-btn');
                const cancelCropBtn = document.getElementById('cancel-crop-btn');
                const closeCropBtn = document.getElementById('close-crop-btn');
                const errorMessage = document.getElementById('error-message');
                
                // State variables
                let selectedFiles = [];
                let cropper = null;
                let currentCropIndex = -1;
                
                // Initialize file input change handler
                fileInput.addEventListener('change', function(e) {
                    const files = Array.from(e.target.files);
                    if (files.length > 0) {
                        // Initialize selected files with crop data
                        const newFiles = files.map(file => ({
                            file: file,
                            cropData: null,
                            cropped: false
                        }));
                        selectedFiles = [...selectedFiles, ...newFiles];
                        updateFileList();
                    }
                });
                
                // Update the file list display
                function updateFileList() {
                    fileList.innerHTML = '';
                    fileList.style.display = selectedFiles.length > 0 ? 'block' : 'none';
                    
                    selectedFiles.forEach((fileData, index) => {
                        const fileItem = document.createElement('div');
                        fileItem.className = 'file-item';
                        
                        const fileName = document.createElement('div');
                        fileName.className = 'file-name';
                        fileName.innerHTML = `
                            <span class="material-icons">description</span>
                            ${fileData.file.name}
                            ${fileData.cropped ? '<span class="file-badge">Cropped</span>' : ''}
                        `;
                        
                        const actions = document.createElement('div');
                        actions.className = 'file-actions';
                        actions.innerHTML = `
                            <button class="action-btn crop" data-index="${index}" title="Crop Image">
                                <span class="material-icons">crop</span>
                            </button>
                            <button class="action-btn remove" data-index="${index}" title="Remove File">
                                <span class="material-icons">delete</span>
                            </button>
                        `;
                        
                        fileItem.appendChild(fileName);
                        fileItem.appendChild(actions);
                        fileList.appendChild(fileItem);
                    });
                    
                    // Add event listeners to action buttons
                    document.querySelectorAll('.action-btn.crop').forEach(btn => {
                        btn.addEventListener('click', function() {
                            const index = parseInt(this.getAttribute('data-index'));
                            openCropModal(index);
                        });
                    });
                    
                    document.querySelectorAll('.action-btn.remove').forEach(btn => {
                        btn.addEventListener('click', function() {
                            const index = parseInt(this.getAttribute('data-index'));
                            selectedFiles.splice(index, 1);
                            updateFileList();
                        });
                    });
                }
                
                // Open crop modal and initialize image for cropping
                function openCropModal(index) {
                    try {
                        currentCropIndex = index;
                        const fileData = selectedFiles[index];
                        
                        // Reset error message
                        errorMessage.style.display = 'none';
                        
                        // Show modal
                        cropModal.style.display = 'block';
                        document.body.style.overflow = 'hidden'; // Prevent scrolling
                        
                        // Reset image and destroy any existing cropper
                        if (cropper) {
                            cropper.destroy();
                            cropper = null;
                        }
                        
                        // Create URL for the image and set it as the source
                        const imageUrl = URL.createObjectURL(fileData.file);
                        cropImage.src = imageUrl;
                        
                        // Once the image is loaded, initialize the cropper
                        cropImage.onload = function() {
                            console.log(`Image loaded: ${cropImage.width}x${cropImage.height}`);
                            
                            // Initialize cropper after a short delay
                            setTimeout(() => {
                                cropper = new Cropper(cropImage, {
                                    viewMode: 1,
                                    dragMode: 'crop',
                                    autoCropArea: 0.8,
                                    responsive: true,
                                    restore: false,
                                    guides: true,
                                    center: true,
                                    highlight: true,
                                    cropBoxMovable: true,
                                    cropBoxResizable: true,
                                    ready: function() {
                                        console.log('Cropper initialized successfully');
                                    }
                                });
                            }, 200);
                        };
                        
                        // Handle image loading errors
                        cropImage.onerror = function() {
                            errorMessage.textContent = 'Failed to load the image. Please try again with a different image.';
                            errorMessage.style.display = 'block';
                            console.error('Failed to load image');
                        };
                    } catch (error) {
                        console.error('Error opening crop modal:', error);
                        errorMessage.textContent = 'An error occurred while preparing the image for cropping.';
                        errorMessage.style.display = 'block';
                    }
                }
                
                // Apply crop to the current image
                cropBtn.addEventListener('click', function() {
                    if (!cropper || currentCropIndex < 0) {
                        return;
                    }
                    
                    try {
                        // Get crop data
                        const cropData = cropper.getData(true); // rounded values
                        
                        // Store crop data with file
                        selectedFiles[currentCropIndex].cropData = {
                            x: cropData.x,
                            y: cropData.y,
                            width: cropData.width,
                            height: cropData.height
                        };
                        selectedFiles[currentCropIndex].cropped = true;
                        
                        // Close crop modal
                        closeCropModal();
                        
                        // Update file list
                        updateFileList();
                    } catch (error) {
                        console.error('Error applying crop:', error);
                        errorMessage.textContent = 'Failed to apply crop. Please try again.';
                        errorMessage.style.display = 'block';
                    }
                });
                
                // Close crop modal without applying
                cancelCropBtn.addEventListener('click', closeCropModal);
                closeCropBtn.addEventListener('click', closeCropModal);
                
                // Close crop modal and clean up
                function closeCropModal() {
                    cropModal.style.display = 'none';
                    document.body.style.overflow = ''; // Restore scrolling
                    if (cropper) {
                        cropper.destroy();
                        cropper = null;
                    }
                    currentCropIndex = -1;
                }
                
                // Process and upload images
                uploadBtn.addEventListener('click', async () => {
                    if (selectedFiles.length === 0) {
                        alert('Please select at least one image file');
                        return;
                    }
                    
                    // Show loading spinner
                    loading.style.display = 'block';
                    results.style.display = 'none';
                    results.innerHTML = '';
                    
                    try {
                        // Process each file individually
                        const processedResults = [];
                        
                        for (const fileData of selectedFiles) {
                            const formData = new FormData();
                            formData.append('file', fileData.file);
                            
                            // Add crop data if available
                            if (fileData.cropData) {
                                formData.append('crop_data', JSON.stringify(fileData.cropData));
                            }
                            
                            const response = await fetch('/upload_image/', {
                                method: 'POST',
                                body: formData
                            });
                            
                            if (response.ok) {
                                const data = await response.json();
                                processedResults.push({
                                    filename: fileData.file.name,
                                    data: data,
                                    cropped: fileData.cropped
                                });
                            } else {
                                processedResults.push({
                                    filename: fileData.file.name,
                                    error: `Error processing file: ${response.statusText}`
                                });
                            }
                        }
                        
                        // Hide loading spinner
                        loading.style.display = 'none';
                        
                        // Display results
                        results.style.display = 'block';
                        
                        // Create results for each prescription
                        processedResults.forEach((result, index) => {
                            const resultItem = document.createElement('div');
                            resultItem.className = 'result-item';
                            
                            if (result.error) {
                                resultItem.innerHTML = `
                                    <div class="result-header">
                                        ${result.filename}
                                    </div>
                                    <div class="result-body">
                                        <div class="result-content">
                                            <div class="error-message" style="display: block;">
                                                ${result.error}
                                            </div>
                                        </div>
                                    </div>
                                `;
                            } else {
                                const resultId = `result-${index}`;
                                
                                resultItem.innerHTML = `
                                    <div class="result-header">
                                        ${result.filename} ${result.cropped ? '<span class="file-badge">Cropped</span>' : ''}
                                    </div>
                                    <div class="result-body">
                                        <div class="result-tabs">
                                            <div class="result-tab active" data-target="${resultId}-text">
                                                Extracted Text
                                            </div>
                                            <div class="result-tab" data-target="${resultId}-analysis">
                                                Analysis
                                            </div>
                                        </div>
                                        <div class="result-content">
                                            <div id="${resultId}-text" class="result-section active">
                                                <div class="extraction-text">${result.data.extracted_text || 'No text extracted'}</div>
                                            </div>
                                            <div id="${resultId}-analysis" class="result-section">
                                                <div class="analysis-grid">
                                                    <div class="analysis-card">
                                                        <h4 class="analysis-title">
                                                            <span class="material-icons">medication</span>
                                                            Medications
                                                        </h4>
                                                        <ul class="analysis-list">
                                                            ${renderAnalysisList(result.data.analysis.medications)}
                                                        </ul>
                                                    </div>
                                                    <div class="analysis-card">
                                                        <h4 class="analysis-title">
                                                            <span class="material-icons">straighten</span>
                                                            Dosages
                                                        </h4>
                                                        <ul class="analysis-list">
                                                            ${renderAnalysisList(result.data.analysis.dosages)}
                                                        </ul>
                                                    </div>
                                                    <div class="analysis-card">
                                                        <h4 class="analysis-title">
                                                            <span class="material-icons">schedule</span>
                                                            Frequencies
                                                        </h4>
                                                        <ul class="analysis-list">
                                                            ${renderAnalysisList(result.data.analysis.frequencies)}
                                                        </ul>
                                                    </div>
                                                    <div class="analysis-card">
                                                        <h4 class="analysis-title">
                                                            <span class="material-icons">calendar_today</span>
                                                            Durations
                                                        </h4>
                                                        <ul class="analysis-list">
                                                            ${renderAnalysisList(result.data.analysis.durations)}
                                                        </ul>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                `;
                            }
                            
                            results.appendChild(resultItem);
                        });
                        
                        // Add event listeners to tabs
                        document.querySelectorAll('.result-tab').forEach(tab => {
                            tab.addEventListener('click', function() {
                                const targetId = this.getAttribute('data-target');
                                const tabsContainer = this.closest('.result-tabs');
                                const contentContainer = this.closest('.result-body').querySelector('.result-content');
                                
                                // Remove active class from all tabs and sections
                                tabsContainer.querySelectorAll('.result-tab').forEach(t => {
                                    t.classList.remove('active');
                                });
                                contentContainer.querySelectorAll('.result-section').forEach(s => {
                                    s.classList.remove('active');
                                });
                                
                                // Add active class to clicked tab and corresponding section
                                this.classList.add('active');
                                document.getElementById(targetId).classList.add('active');
                            });
                        });
                        
                    } catch (error) {
                        console.error('Error:', error);
                        loading.style.display = 'none';
                        alert('Error processing the images. Please try again.');
                    }
                });
                
                // Helper function to render analysis list
                function renderAnalysisList(items) {
                    if (!items || items.length === 0) {
                        return '<li class="empty-message">None detected</li>';
                    }
                    
                    return items.map(item => `<li>${item}</li>`).join('');
                }
                
                // Close modal when clicking outside
                window.addEventListener('click', function(event) {
                    if (event.target === cropModal) {
                        closeCropModal();
                    }
                });
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/upload_image/")
async def upload_image(
    file: UploadFile = File(...),
    crop_data: Optional[str] = Form(None)
):
    """Handle image upload and process it with optional cropping."""
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Apply cropping if crop data is provided
        if crop_data:
            try:
                crop_dict = json.loads(crop_data)
                image = crop_image(image, crop_dict)
            except json.JSONDecodeError:
                logger.warning("Invalid crop data format")
            except Exception as e:
                logger.error(f"Error applying crop: {str(e)}")
        
        # Extract text from image
        extracted_text = extract_text_from_image(image)
        
        if not extracted_text:
            return {
                "error": "No text could be extracted from the image",
                "extracted_text": "",
                "analysis": {}
            }
        
        # Analyze the prescription
        analysis = analyze_prescription(extracted_text)
        
        return {
            "extracted_text": extracted_text,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_multiple_images/")
async def upload_multiple_images(files: List[UploadFile] = File(...)):
    """Handle multiple image uploads and process them."""
    results = []
    
    for file in files:
        try:
            # Read image file
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                results.append({
                    "filename": file.filename,
                    "error": "Invalid image file"
                })
                continue
            
            # Extract text from image
            extracted_text = extract_text_from_image(image)
            
            if not extracted_text:
                results.append({
                    "filename": file.filename,
                    "error": "No text could be extracted from the image",
                    "extracted_text": "",
                    "analysis": {}
                })
                continue
            
            # Analyze the prescription
            analysis = analyze_prescription(extracted_text)
            
            results.append({
                "filename": file.filename,
                "extracted_text": extracted_text,
                "analysis": analysis
            })
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return results

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 