# Drug Interaction Detection System
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import os
import requests  # ✅ Added for API calls

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Hugging Face API Config
# API Keys Configuration with fallback handling
def get_working_hf_token_fallback():
    """Fallback function to get working HF token"""
    tokens = [
        "HF_TOKEN",
        "HF_TOKEN",
        "HF_TOKEN",
        os.getenv("HF_API_TOKEN", "")
    ]
    for token in tokens:
        if token and token.strip() and not token.startswith("your_"):
            return token
    return "HF_TOKEN"  # Default fallback

try:
    # Try to import from centralized config
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from api_keys_config import get_working_hf_token  # type: ignore
    HF_API_TOKEN = get_working_hf_token()
    HF_API_TOKEN_READ = get_working_hf_token()
    print("✅ Using centralized API configuration")

except (ImportError, ModuleNotFoundError):
    # Fallback to embedded API keys
    print("⚠️ Using fallback API configuration")
    get_working_hf_token = get_working_hf_token_fallback  # Define the function
    HF_API_TOKEN = get_working_hf_token()
    HF_API_TOKEN_READ = get_working_hf_token()

# Specialized Medical Models with API Keys
DDI_MODEL_PRIMARY = "ltmai/Bio_ClinicalBERT_DDI_finetuned"  # Updated to use the specialized DDI model
DDI_MODEL_SECONDARY = "d4data/biomedical-ner-all"
DDI_MODEL_TERTIARY = "bprimal/Drug-Drug-Interaction-Classification"
NER_MODEL_PRIMARY = "d4data/biomedical-ner-all"
NER_MODEL_SECONDARY = "Clinical-AI-Apollo/Medical-NER"
# Update to use the specialized DDI model
HF_DDI_MODEL = DDI_MODEL_PRIMARY  # Use the Bio_ClinicalBERT_DDI_finetuned model
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_DDI_MODEL}"

# Try multiple API keys for better access
def get_hf_headers():
    """Get HF headers with fallback API keys"""
    api_keys = [HF_API_TOKEN, HF_API_TOKEN_READ]
    for key in api_keys:
        if key and key.strip():
            return {"Authorization": f"Bearer {key}"}
    return {"Authorization": f"Bearer {HF_API_TOKEN}"}  # Fallback

HF_HEADERS = get_hf_headers()

def query_hf_ddi_api(drug1: str, drug2: str) -> Optional[Dict[str, Any]]:
    """
    Call Hugging Face hosted model to check interaction between two drugs.
    """
    try:
        payload = {"inputs": f"{drug1} [SEP] {drug2}"}
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result  # HF model's raw output
        else:
            logger.error(f"Hugging Face API Error {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"HF API request failed: {str(e)}")
    return None

class InteractionSeverity(Enum):
    CONTRAINDICATED = "contraindicated"
    MAJOR = "major"
    MODERATE = "moderate"
    MINOR = "minor"
    UNKNOWN = "unknown"

@dataclass
class DrugInteraction:
    drug1: str
    drug2: str
    severity: InteractionSeverity
    mechanism: str
    clinical_effect: str
    management: str
    evidence_level: str
    risk_score: int

@dataclass
class DrugInfo:
    name: str
    generic_name: str
    drug_class: str
    mechanism_of_action: str
    contraindications: List[str]
    warnings: List[str]

class DrugInteractionDetector:
    def __init__(self):
        self.interaction_database = self.load_interaction_database()
        self.drug_database = self.load_drug_database()
        self.interaction_rules = self.setup_interaction_rules()
    
    def load_interaction_database(self) -> Dict[str, Any]:
        # ... your existing function (unchanged) ...
        interactions = {
            # ... your hardcoded interactions ...
        }
        return interactions

    def load_drug_database(self) -> Dict[str, DrugInfo]:
        # ... unchanged ...
        return {
            # ... unchanged drug info ...
        }

    def setup_interaction_rules(self) -> Dict[str, Any]:
        # ... unchanged ...
        return {
            # ... unchanged rules ...
        }

    def normalize_drug_name(self, drug_name: str) -> str:
        """Normalize drug name for consistent matching"""
        if not drug_name:
            return ""

        # Convert to lowercase and strip whitespace
        normalized = drug_name.lower().strip()

        # Remove common suffixes and prefixes
        suffixes_to_remove = [
            ' tablet', ' tablets', ' capsule', ' capsules',
            ' mg', ' mcg', ' g', ' ml', ' injection',
            ' oral', ' iv', ' im', ' topical'
        ]

        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()

        # Handle common drug name variations
        drug_mappings = {
            'acetaminophen': 'paracetamol',
            'tylenol': 'paracetamol',
            'advil': 'ibuprofen',
            'motrin': 'ibuprofen',
            'coumadin': 'warfarin',
            'aspirin': 'acetylsalicylic acid'
        }

        return drug_mappings.get(normalized, normalized)

    def detect_interactions(self, drug_list: List[str]) -> List[DrugInteraction]:
        interactions = []
        normalized_drugs = [self.normalize_drug_name(drug) for drug in drug_list]
        
        for i, drug1 in enumerate(normalized_drugs):
            for drug2 in normalized_drugs[i+1:]:
                # ✅ First, check with Hugging Face API
                hf_result = query_hf_ddi_api(drug1, drug2)
                if hf_result:
                    # Handle different HF API response formats
                    predicted_label = 'unknown'
                    try:
                        if isinstance(hf_result, list) and len(hf_result) > 0:
                            first_result = hf_result[0]  # type: ignore
                            if isinstance(first_result, dict):
                                predicted_label = str(first_result.get('label', 'UNKNOWN')).lower()
                        elif isinstance(hf_result, dict):
                            predicted_label = str(hf_result.get('label', 'UNKNOWN')).lower()
                    except (IndexError, KeyError, TypeError):
                        predicted_label = 'unknown'
                    severity_map = {
                        "contraindicated": InteractionSeverity.CONTRAINDICATED,
                        "major": InteractionSeverity.MAJOR,
                        "moderate": InteractionSeverity.MODERATE,
                        "minor": InteractionSeverity.MINOR
                    }
                    severity = severity_map.get(predicted_label, InteractionSeverity.UNKNOWN)

                    interactions.append(
                        DrugInteraction(
                            drug1=drug1,
                            drug2=drug2,
                            severity=severity,
                            mechanism="Predicted by HF DDI model",
                            clinical_effect="See API reference",
                            management="Refer to clinical guidelines",
                            evidence_level="AI-Predicted",
                            risk_score=self.severity_to_risk_score(severity)
                        )
                    )
                    continue  # Skip local DB if API result exists

                # If API didn't return, fall back to local DB
                interaction = self.check_drug_pair_interaction(drug1, drug2)
                if interaction:
                    interactions.append(interaction)
        
        rule_interactions = self.check_rule_based_interactions(normalized_drugs)
        interactions.extend(rule_interactions)
        
        interactions.sort(key=lambda x: (x.severity.value, -x.risk_score))
        return interactions

    def severity_to_risk_score(self, severity: InteractionSeverity) -> int:
        """Convert severity to risk score"""
        severity_scores = {
            InteractionSeverity.MINOR: 3,
            InteractionSeverity.MODERATE: 5,
            InteractionSeverity.MAJOR: 8,
            InteractionSeverity.CONTRAINDICATED: 10
        }
        return severity_scores.get(severity, 5)

    def check_drug_pair_interaction(self, drug1: str, drug2: str) -> Optional[DrugInteraction]:
        """Check for interactions between two specific drugs"""
        # Simple example database - in practice, this would be a comprehensive database
        known_interactions = {
            ('warfarin', 'acetylsalicylic acid'): {
                'severity': InteractionSeverity.MAJOR,
                'mechanism': 'Increased bleeding risk due to anticoagulant and antiplatelet effects',
                'clinical_effect': 'Significantly increased risk of bleeding',
                'management': 'Monitor INR closely, consider dose adjustment',
                'evidence_level': 'Well-documented'
            },
            ('warfarin', 'ibuprofen'): {
                'severity': InteractionSeverity.MAJOR,
                'mechanism': 'NSAIDs can increase bleeding risk with anticoagulants',
                'clinical_effect': 'Increased bleeding risk',
                'management': 'Avoid concurrent use or monitor closely',
                'evidence_level': 'Well-documented'
            }
        }

        # Check both directions
        pair1 = (drug1, drug2)
        pair2 = (drug2, drug1)

        interaction_data = known_interactions.get(pair1) or known_interactions.get(pair2)

        if interaction_data:
            return DrugInteraction(
                drug1=drug1,
                drug2=drug2,
                severity=interaction_data['severity'],
                mechanism=interaction_data['mechanism'],
                clinical_effect=interaction_data['clinical_effect'],
                management=interaction_data['management'],
                evidence_level=interaction_data['evidence_level'],
                risk_score=self.severity_to_risk_score(interaction_data['severity'])
            )

        return None

    def check_rule_based_interactions(self, drug_list: List[str]) -> List[DrugInteraction]:
        """Check for rule-based interactions (e.g., drug classes)"""
        interactions = []

        # Example: Check for multiple NSAIDs
        nsaids = ['ibuprofen', 'naproxen', 'diclofenac', 'celecoxib']
        found_nsaids = [drug for drug in drug_list if drug in nsaids]

        if len(found_nsaids) > 1:
            for i, drug1 in enumerate(found_nsaids):
                for drug2 in found_nsaids[i+1:]:
                    interactions.append(DrugInteraction(
                        drug1=drug1,
                        drug2=drug2,
                        severity=InteractionSeverity.MODERATE,
                        mechanism='Multiple NSAIDs increase risk of GI and renal toxicity',
                        clinical_effect='Increased risk of gastrointestinal bleeding and kidney damage',
                        management='Avoid concurrent use of multiple NSAIDs',
                        evidence_level='Rule-based',
                        risk_score=6
                    ))

        return interactions

    def get_interaction_summary(self, interactions: List[DrugInteraction]) -> Dict[str, Any]:
        """Get summary statistics for interactions"""
        if not interactions:
            return {
                'total_interactions': 0,
                'max_risk_score': 0,
                'severity_counts': {},
                'highest_severity': None,
                'recommendations': ["✅ No significant drug interactions detected"]
            }

        severity_counts = {}
        for interaction in interactions:
            severity_name = interaction.severity.name
            severity_counts[severity_name] = severity_counts.get(severity_name, 0) + 1

        max_risk_score = max(interaction.risk_score for interaction in interactions)
        highest_severity = max(interactions, key=lambda x: x.severity.value).severity

        # Generate recommendations based on interactions
        recommendations = []
        if max_risk_score >= 8:
            recommendations.append("⚠️ HIGH RISK: Consult healthcare provider immediately")
        if max_risk_score >= 5:
            recommendations.append("📋 Monitor patient closely for adverse effects")
        if any(interaction.severity == InteractionSeverity.CONTRAINDICATED for interaction in interactions):
            recommendations.append("🚫 CONTRAINDICATED: Do not use these drugs together")

        if not recommendations:
            recommendations.append("✅ Low risk interactions - standard monitoring recommended")

        return {
            'total_interactions': len(interactions),
            'max_risk_score': max_risk_score,
            'severity_counts': severity_counts,
            'highest_severity': highest_severity.name,
            'recommendations': recommendations
        }

    # ... rest of your class unchanged (check_drug_pair_interaction, rules, summary, etc.) ...

# Example usage and testing
if __name__ == "__main__":
    detector = DrugInteractionDetector()
    test_combinations = [
        ["Warfarin", "Aspirin"],
        ["Metformin", "Lisinopril"],
        ["Simvastatin", "Clarithromycin"],
        ["Warfarin", "Ibuprofen", "Metformin"],
        ["Sertraline", "Tramadol"]
    ]
    print("🧪 Testing Drug Interaction Detection:")
    print("=" * 60)
    for i, drugs in enumerate(test_combinations, 1):
        print(f"\nTest {i}: {', '.join(drugs)}")
        interactions = detector.detect_interactions(drugs)
        summary = detector.get_interaction_summary(interactions)
        print(f"Total interactions: {summary['total_interactions']}")
        print(f"Max risk score: {summary['max_risk_score']}")
        for interaction in interactions:
            print(f"  ⚠️ {interaction.drug1} + {interaction.drug2}")
            print(f"     Severity: {interaction.severity.value}")
            print(f"     Effect: {interaction.clinical_effect}")
            print(f"     Management: {interaction.management}")
            print("     ---")
        print("Recommendations:")
        for rec in summary['recommendations']:
            print(f"  {rec}")
        print("-" * 40)
