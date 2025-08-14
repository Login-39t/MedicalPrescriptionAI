"""
Fix remaining Config attribute accesses
"""

import re

def fix_remaining_config():
    """Fix all remaining Config.ATTRIBUTE accesses"""
    
    with open('enhanced_drug_flask_ai.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define all remaining replacements
    replacements = [
        ('Config.IBM_API_KEY', "getattr(Config, 'IBM_API_KEY', '')"),
        ('Config.HF_API_TOKEN', "getattr(Config, 'HF_API_TOKEN', '')"),
        ('Config.FASTAPI_URL', "getattr(Config, 'FASTAPI_URL', 'http://localhost:8001')"),
        ('Config.FALLBACK_URL', "getattr(Config, 'FALLBACK_URL', 'http://localhost:8000')"),
        ('Config.GRANITE_MEDICAL_MODEL', "getattr(Config, 'GRANITE_MEDICAL_MODEL', 'ibm-granite/granite-3.2-2b-instruct')"),
        ('Config.IBM_WATSONX_MODEL_ID', "getattr(Config, 'IBM_WATSONX_MODEL_ID', 'ibm/granite-3-2b-instruct')"),
    ]
    
    # Apply replacements
    for old, new in replacements:
        content = content.replace(old, new)
    
    # Write back
    with open('enhanced_drug_flask_ai.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed all remaining Config attribute accesses")

if __name__ == "__main__":
    fix_remaining_config()
