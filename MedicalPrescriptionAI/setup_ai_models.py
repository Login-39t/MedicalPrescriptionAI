#!/usr/bin/env python3
"""
Setup script for Enhanced Drug Analysis System with AI Models
This script helps you install and configure HuggingFace and IBM Granite models
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_step(step, text):
    """Print a formatted step"""
    print(f"\n[{step}] {text}")

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print_step("1", "Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print_step("2", "Installing dependencies...")
    
    # Core dependencies
    core_deps = [
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "flask>=3.0.0",
        "requests>=2.31.0",
        "pydantic>=2.5.0"
    ]
    
    print("Installing core dependencies...")
    for dep in core_deps:
        if not run_command(f"pip install {dep}"):
            print(f"❌ Failed to install {dep}")
            return False
    
    # AI/ML dependencies
    ai_deps = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "tokenizers>=0.15.0",
        "accelerate>=0.24.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0"
    ]
    
    print("Installing AI/ML dependencies...")
    for dep in ai_deps:
        if not run_command(f"pip install {dep}"):
            print(f"⚠️ Warning: Failed to install {dep}")
    
    # Optional dependencies
    optional_deps = [
        "spacy>=3.7.0",
        "nltk>=3.8.0",
        "plotly>=5.17.0",
        "python-dotenv>=1.0.0"
    ]
    
    print("Installing optional dependencies...")
    for dep in optional_deps:
        run_command(f"pip install {dep}")
    
    print("✅ Dependencies installation completed")
    return True

def setup_environment_variables():
    """Setup environment variables for API keys"""
    print_step("3", "Setting up environment variables...")
    
    env_file = Path(".env")
    env_content = []
    
    # IBM API Key
    print("\n🔑 IBM Watson/Granite Configuration:")
    print("To use IBM Granite models, you need:")
    print("1. IBM Cloud account")
    print("2. Watson Machine Learning service")
    print("3. API key and project ID")
    
    ibm_api_key = input("Enter your IBM API Key (or press Enter to skip): ").strip()
    if ibm_api_key:
        env_content.append(f"IBM_API_KEY={ibm_api_key}")
        
        ibm_project_id = input("Enter your IBM Project ID: ").strip()
        if ibm_project_id:
            env_content.append(f"IBM_PROJECT_ID={ibm_project_id}")
    
    # HuggingFace Token
    print("\n🤗 HuggingFace Configuration:")
    print("For enhanced model access, you can provide a HuggingFace token")
    print("Get your token from: https://huggingface.co/settings/tokens")
    
    hf_token = input("Enter your HuggingFace token (or press Enter to skip): ").strip()
    if hf_token:
        env_content.append(f"HF_API_TOKEN={hf_token}")
    
    # Write .env file
    if env_content:
        with open(env_file, "w") as f:
            f.write("\n".join(env_content))
        print(f"✅ Environment variables saved to {env_file}")
    else:
        print("⚠️ No environment variables configured")
    
    return True

def download_huggingface_models():
    """Download HuggingFace models for medical analysis"""
    print_step("4", "Configuring HuggingFace models for API-only usage...")
    
    models_to_download = [
        "dmis-lab/biobert-base-cased-v1.1",
        "emilyalsentzer/Bio_ClinicalBERT", 
        "alvaroalon2/biobert_diseases_ner"
    ]
    
    print("✅ Skipping model downloads to save disk space and memory")
    print("💡 Models will be accessed via HuggingFace API calls when needed")
    print("💡 This prevents your laptop from downloading several GB of model data")
    
    return True

def setup_spacy_models():
    """Setup spaCy models for medical NLP"""
    print_step("5", "Setting up spaCy models...")
    
    print("✅ Skipping spaCy model downloads to save disk space")
    print("💡 Basic NLP functionality will use built-in capabilities")
    print("💡 Advanced features will use HuggingFace API calls")
    
    return True

def create_config_file():
    """Create configuration file"""
    print_step("6", "Creating configuration file...")
    
    config = {
        "ai_models": {
            "biobert_model": "dmis-lab/biobert-base-cased-v1.1",
            "clinical_bert": "emilyalsentzer/Bio_ClinicalBERT",
            "drug_ner_model": "alvaroalon2/biobert_diseases_ner",
            "device": "cuda" if torch_available() else "cpu"
        },
        "ibm_granite": {
            "model_id": "ibm/granite-13b-chat-v2",
            "url": "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation"
        },
        "api_endpoints": {
            "enhanced_backend": "http://localhost:8001",
            "fallback_backend": "http://localhost:8000",
            "frontend": "http://localhost:5002"
        },
        "features": {
            "ai_drug_extraction": True,
            "interaction_prediction": True,
            "safety_assessment": True,
            "personalized_recommendations": True
        }
    }
    
    with open("ai_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration file created: ai_config.json")
    return True

def torch_available():
    """Check if PyTorch is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def test_installation():
    """Test the installation"""
    print_step("7", "Testing installation...")
    
    # Test imports
    try:
        import fastapi
        import transformers
        import torch
        print("✅ Core imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test model loading (basic)
    try:
        from transformers import AutoTokenizer
        print("✅ HuggingFace transformers available (API mode)")
        print("💡 Models will be accessed via API calls, not local downloads")
    except Exception as e:
        print(f"⚠️ HuggingFace test failed: {e}")
    
    # Test device
    if torch_available():
        print("✅ CUDA available for GPU acceleration")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    return True

def print_next_steps():
    """Print next steps for the user"""
    print_header("🎉 Setup Complete!")
    
    print("""
Next steps:

1. 📚 Start the Enhanced AI Backend:
   python enhanced_drug_analysis_hf.py

2. 🌐 Start the Enhanced Frontend:
   python enhanced_drug_flask_ai.py

3. 🔗 Access the application:
   http://localhost:5002

4. 📖 API Documentation:
   http://localhost:8001/docs

5. 🔧 Configuration:
   - Edit ai_config.json for custom settings
   - Update .env for API keys
   - Check logs for any issues

Features available:
✅ AI-powered drug extraction (HuggingFace BioBERT)
✅ IBM Granite analysis integration
✅ Enhanced safety assessment
✅ Smart interaction detection
✅ Personalized recommendations

For help and documentation:
- Check the README files
- Visit the API docs at /docs
- Review the configuration files
""")

def main():
    """Main setup function"""
    print_header("🚀 Enhanced Drug Analysis System Setup")
    print("This script will help you set up the AI-powered drug analysis system")
    print("with HuggingFace and IBM Granite integration.")
    
    # Check requirements
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Setup environment
    setup_environment_variables()
    
    # Download models
    download_huggingface_models()
    
    # Setup spaCy
    setup_spacy_models()
    
    # Create config
    create_config_file()
    
    # Test installation
    test_installation()
    
    # Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Setup completed successfully!")
        else:
            print("\n❌ Setup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
