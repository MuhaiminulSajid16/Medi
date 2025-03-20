def check_imports():
    dependencies = {
        'FastAPI': 'fastapi',
        'Uvicorn': 'uvicorn',
        'PIL (Pillow)': 'PIL',
        'PyTorch': 'torch',
        'OpenCV': 'cv2',
        'Transformers': 'transformers',
        'EasyOCR': 'easyocr',
        'Datasets': 'datasets',
        'Python-dotenv': 'dotenv',
        'Pydantic': 'pydantic',
        'NumPy': 'numpy'
    }
    
    print("Checking dependencies...")
    print("-" * 50)
    
    all_installed = True
    
    for name, package in dependencies.items():
        try:
            __import__(package)
            print(f"✅ {name:<20} is installed")
        except ImportError as e:
            print(f"❌ {name:<20} is NOT installed - Error: {str(e)}")
            all_installed = False
    
    print("-" * 50)
    if all_installed:
        print("All dependencies are installed successfully! 🎉")
    else:
        print("Some dependencies are missing. Please install them using pip install -r requirements.txt")

if __name__ == "__main__":
    check_imports() 