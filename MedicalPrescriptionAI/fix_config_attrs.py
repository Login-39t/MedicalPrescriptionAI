"""
Script to fix Config attribute accesses in enhanced_drug_flask_ai.py
"""

import re

def fix_config_attributes():
    """Fix all Config.ATTRIBUTE accesses to use getattr"""
    
    # Read the file
    with open('enhanced_drug_flask_ai.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define replacements for common Config attributes
    replacements = {
        'Config.HF_API_TOKEN': "getattr(Config, 'HF_API_TOKEN', '')",
        'Config.HF_API_TOKEN_READ': "getattr(Config, 'HF_API_TOKEN_READ', '')",
        'Config.DDI_MODEL_PRIMARY': "getattr(Config, 'DDI_MODEL_PRIMARY', 'ltmai/Bio_ClinicalBERT_DDI_finetuned')",
        'Config.DDI_MODEL_SECONDARY': "getattr(Config, 'DDI_MODEL_SECONDARY', 'bprimal/Drug-Drug-Interaction-Classification')",
        'Config.NER_MODEL_PRIMARY': "getattr(Config, 'NER_MODEL_PRIMARY', 'd4data/biomedical-ner-all')",
        'Config.NER_MODEL_SECONDARY': "getattr(Config, 'NER_MODEL_SECONDARY', 'Clinical-AI-Apollo/Medical-NER')",
        'Config.NER_MODEL_TERTIARY': "getattr(Config, 'NER_MODEL_TERTIARY', 'alvaroalon2/biobert_diseases_ner')",
        'Config.GRANITE_MEDICAL_MODEL': "getattr(Config, 'GRANITE_MEDICAL_MODEL', 'ibm-granite/granite-3.2-2b-instruct')",
        'Config.IBM_API_KEY': "getattr(Config, 'IBM_API_KEY', '')",
        'Config.IBM_PROJECT_ID': "getattr(Config, 'IBM_PROJECT_ID', '')",
        'Config.IBM_WATSONX_MODEL_ID': "getattr(Config, 'IBM_WATSONX_MODEL_ID', 'ibm/granite-3-2b-instruct')",
        'Config.IBM_WATSONX_URL': "getattr(Config, 'IBM_WATSONX_URL', 'https://us-south.ml.cloud.ibm.com')",
        'Config.IBM_WATSONX_VERSION': "getattr(Config, 'IBM_WATSONX_VERSION', '2023-05-29')",
        'Config.FASTAPI_URL': "getattr(Config, 'FASTAPI_URL', 'http://localhost:8001')",
        'Config.FALLBACK_URL': "getattr(Config, 'FALLBACK_URL', 'http://localhost:8000')",
    }
    
    # Apply replacements
    for old_attr, new_attr in replacements.items():
        content = content.replace(old_attr, new_attr)
    
    # Handle conditional expressions like "Config.HF_API_TOKEN if Config.HF_API_TOKEN else None"
    # Replace with just the getattr call since it already handles the default
    conditional_patterns = [
        (r"getattr\(Config, 'HF_API_TOKEN', ''\) if getattr\(Config, 'HF_API_TOKEN', ''\) else None", 
         "getattr(Config, 'HF_API_TOKEN', None)"),
        (r"getattr\(Config, 'HF_API_TOKEN_READ', ''\) if getattr\(Config, 'HF_API_TOKEN_READ', ''\) else getattr\(Config, 'HF_API_TOKEN', ''\)", 
         "getattr(Config, 'HF_API_TOKEN_READ', None) or getattr(Config, 'HF_API_TOKEN', None)"),
    ]
    
    for pattern, replacement in conditional_patterns:
        content = re.sub(pattern, replacement, content)
    
    # Write the file back
    with open('enhanced_drug_flask_ai.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed all Config attribute accesses")

if __name__ == "__main__":
    fix_config_attributes()
