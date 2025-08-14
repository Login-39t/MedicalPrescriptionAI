"""
API Keys Configuration for Medical AI Models
============================================

This file contains all the API keys and model configurations needed for the 
specialized medical models to work properly.

IMPORTANT: Keep this file secure and never commit it to public repositories!
"""

import os

# =============================================================================
# HUGGING FACE API KEYS
# =============================================================================

# Primary HuggingFace API Keys (with write access)
HF_API_TOKEN_PRIMARY = "HF_TOKEN"
HF_API_TOKEN_SECONDARY = "HF_TOKEN"
HF_API_TOKEN_TERTIARY = "HF_TOKEN"

# Read-only HuggingFace API Keys (for model downloads)
HF_API_TOKEN_READ_1 = "HF_TOKEN"
HF_API_TOKEN_READ_2 = "HF_TOKEN"

# =============================================================================
# SPECIALIZED MEDICAL MODELS CONFIGURATION
# =============================================================================

# Drug-Drug Interaction (DDI) Models
DDI_MODELS = {
    "primary": {
        "model_id": "d4data/biomedical-ner-all",
        "description": "Biomedical NER model adapted for drug-drug interactions",
        "api_key": HF_API_TOKEN_PRIMARY,
        "model_type": "ner_based_ddi",
        "input_format": "[DRUG1] and [DRUG2]"
    },
    "secondary": {
        "model_id": "ltmai/Bio_ClinicalBERT_DDI_finetuned",
        "description": "Clinical BERT fine-tuned for drug-drug interactions",
        "api_key": HF_API_TOKEN_SECONDARY,
        "model_type": "classification",
        "input_format": "[DRUG1] [SEP] [DRUG2]"
    },
    "tertiary": {
        "model_id": "bprimal/Drug-Drug-Interaction-Classification",
        "description": "Alternative DDI classification model",
        "api_key": HF_API_TOKEN_TERTIARY,
        "model_type": "classification",
        "input_format": "[DRUG1] and [DRUG2]"
    }
}

# Medical Named Entity Recognition (NER) Models
NER_MODELS = {
    "biomedical": {
        "model_id": "d4data/biomedical-ner-all",
        "description": "Comprehensive biomedical NER model",
        "api_key": HF_API_TOKEN_READ_1,
        "model_type": "token_classification",
        "entities": ["DRUG", "DISEASE", "CHEMICAL", "GENE"]
    },
    "clinical": {
        "model_id": "Clinical-AI-Apollo/Medical-NER",
        "description": "Clinical medical NER model",
        "api_key": HF_API_TOKEN_READ_2,
        "model_type": "token_classification", 
        "entities": ["MEDICATION", "CONDITION", "DOSAGE"]
    },
    "disease_focused": {
        "model_id": "alvaroalon2/biobert_diseases_ner",
        "description": "Disease-focused BERT NER model",
        "api_key": HF_API_TOKEN_PRIMARY,
        "model_type": "token_classification",
        "entities": ["DISEASE", "SYMPTOM"]
    }
}

# =============================================================================
# IBM WATSON/GRANITE API CONFIGURATION
# =============================================================================

IBM_CONFIG = {
    "api_key": os.getenv("IBM_API_KEY", ""),  # Set your IBM API key here
    "project_id": os.getenv("IBM_PROJECT_ID", ""),  # Set your IBM project ID here
    "watsonx_url": "https://us-south.ml.cloud.ibm.com",
    "watsonx_version": "2023-05-29",
    "granite_model_id": "ibm/granite-3-2b-instruct"
}

# =============================================================================
# ADDITIONAL AI SERVICE API KEYS (Optional)
# =============================================================================

ADDITIONAL_APIS = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),  # For GPT models if needed
        "models": ["gpt-3.5-turbo", "gpt-4"]
    },
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),  # For Claude models if needed
        "models": ["claude-3-sonnet", "claude-3-haiku"]
    },
    "google": {
        "api_key": os.getenv("GOOGLE_API_KEY", ""),  # For Gemini models if needed
        "models": ["gemini-pro", "gemini-pro-vision"]
    }
}

# =============================================================================
# API ENDPOINTS CONFIGURATION
# =============================================================================

API_ENDPOINTS = {
    "huggingface_inference": "https://api-inference.huggingface.co/models/",
    "huggingface_hub": "https://huggingface.co/",
    "ibm_watsonx": "https://us-south.ml.cloud.ibm.com",
    "openai": "https://api.openai.com/v1/",
    "anthropic": "https://api.anthropic.com/v1/"
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_working_hf_token():
    """Get the first working HuggingFace API token"""
    tokens = [HF_API_TOKEN_PRIMARY, HF_API_TOKEN_SECONDARY, HF_API_TOKEN_READ_1, HF_API_TOKEN_READ_2]
    for token in tokens:
        if token and token.strip() and not token.startswith("your_"):
            return token
    return HF_API_TOKEN_PRIMARY  # Fallback

def get_ddi_model_config(preference="primary"):
    """Get DDI model configuration"""
    return DDI_MODELS.get(preference, DDI_MODELS["primary"])

def get_ner_model_config(preference="biomedical"):
    """Get NER model configuration"""
    return NER_MODELS.get(preference, NER_MODELS["biomedical"])

def validate_api_keys():
    """Validate that API keys are properly configured"""
    issues = []
    
    # Check HuggingFace tokens
    if not get_working_hf_token() or get_working_hf_token().startswith("your_"):
        issues.append("❌ HuggingFace API tokens not properly configured")
    else:
        issues.append("✅ HuggingFace API tokens configured")
    
    # Check IBM configuration
    if not IBM_CONFIG["api_key"]:
        issues.append("⚠️ IBM API key not set (optional for basic functionality)")
    else:
        issues.append("✅ IBM API key configured")
    
    return issues

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("🔑 Medical AI API Keys Configuration")
    print("=" * 50)
    
    # Validate configuration
    validation_results = validate_api_keys()
    for result in validation_results:
        print(result)
    
    print("\n📋 Available Models:")
    print(f"DDI Models: {list(DDI_MODELS.keys())}")
    print(f"NER Models: {list(NER_MODELS.keys())}")
    
    print(f"\n🔗 Working HF Token: {get_working_hf_token()[:10]}...")
    print(f"🤖 Primary DDI Model: {get_ddi_model_config()['model_id']}")
    print(f"🏷️ Primary NER Model: {get_ner_model_config()['model_id']}")
