# NLP Drug Information Extraction System
import re
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass
from drug_config import DRUG_NER_MODEL, DRUG_EXTRACTION_MODEL, DOSAGE_UNITS, FREQUENCY_OPTIONS

# Try to load spacy model, fallback if not available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

@dataclass
class ExtractedDrug:
    name: str
    dosage: Optional[float] = None
    unit: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    duration: Optional[str] = None
    instructions: Optional[str] = None
    confidence: float = 0.0

class DrugNLPExtractor:
    def __init__(self):
        self.setup_models()
        self.setup_patterns()
    
    def setup_models(self):
        """Initialize NLP models"""
        try:
            # HuggingFace NER model for biomedical entities
            self.ner_pipeline = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                tokenizer="d4data/biomedical-ner-all",
                aggregation_strategy="simple"
            )
            print("✅ HuggingFace biomedical NER model loaded")
        except Exception as e:
            print(f"⚠️ Could not load HuggingFace model: {e}")
            self.ner_pipeline = None
        
        # Fallback to rule-based extraction
        self.use_rule_based = True
    
    def setup_patterns(self):
        """Setup regex patterns for drug information extraction"""
        
        # Drug name patterns
        self.drug_patterns = [
            r'\b([A-Z][a-z]+(?:in|ol|ide|ine|ate|ium|ic)?)\b',  # Common drug suffixes
            r'\b([A-Z][a-z]*[A-Z][a-z]*)\b',  # CamelCase drug names
        ]
        
        # Dosage patterns
        self.dosage_patterns = [
            r'(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml|L|units?|IU|tablets?|capsules?|drops?|puffs?)',
            r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml|L|units?|IU)',
            r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml|L)',
        ]
        
        # Frequency patterns
        self.frequency_patterns = [
            r'\b(once|twice|three times?|four times?|1x|2x|3x|4x)\s*(daily|a day|per day)\b',
            r'\bevery\s*(\d+)\s*(hours?|hrs?|h)\b',
            r'\b(q\d+h|qid|tid|bid|qd|prn|ac|pc|hs)\b',
            r'\b(before|after)\s*(meals?|eating|food)\b',
            r'\b(at\s*bedtime|before\s*sleep|hs)\b',
            r'\b(as\s*needed|prn|when\s*required)\b'
        ]
        
        # Route patterns
        self.route_patterns = [
            r'\b(oral|orally|by mouth|po)\b',
            r'\b(intravenous|IV|i\.?v\.?)\b',
            r'\b(intramuscular|IM|i\.?m\.?)\b',
            r'\b(subcutaneous|SC|SQ|s\.?c\.?)\b',
            r'\b(topical|topically|applied)\b',
            r'\b(inhaled|inhalation|nebulized)\b'
        ]
        
        # Duration patterns
        self.duration_patterns = [
            r'\bfor\s*(\d+)\s*(days?|weeks?|months?|years?)\b',
            r'\b(\d+)\s*(day|week|month|year)\s*(course|treatment|therapy)\b',
            r'\bcontinue\s*(for|until)\s*(\d+)\s*(days?|weeks?)\b'
        ]
    
    def extract_drugs_from_text(self, text: str) -> List[ExtractedDrug]:
        """Extract drug information from unstructured text"""
        extracted_drugs = []
        
        # Clean and preprocess text
        text = self.preprocess_text(text)
        
        # Try HuggingFace NER first
        if self.ner_pipeline:
            hf_drugs = self.extract_with_huggingface(text)
            extracted_drugs.extend(hf_drugs)
        
        # Use rule-based extraction as fallback or supplement
        rule_drugs = self.extract_with_rules(text)
        extracted_drugs.extend(rule_drugs)
        
        # Merge and deduplicate
        merged_drugs = self.merge_drug_extractions(extracted_drugs)
        
        return merged_drugs
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better extraction"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize common abbreviations
        abbreviations = {
            'mg.': 'mg',
            'ml.': 'ml',
            'q.d.': 'once daily',
            'b.i.d.': 'twice daily',
            't.i.d.': 'three times daily',
            'q.i.d.': 'four times daily',
            'p.r.n.': 'as needed',
            'a.c.': 'before meals',
            'p.c.': 'after meals',
            'h.s.': 'at bedtime'
        }
        
        for abbrev, full in abbreviations.items():
            text = text.replace(abbrev, full)
        
        return text.strip()
    
    def extract_with_huggingface(self, text: str) -> List[ExtractedDrug]:
        """Extract drugs using HuggingFace biomedical NER"""
        drugs = []
        
        try:
            entities = self.ner_pipeline(text)
            
            for entity in entities:
                if entity['entity_group'] in ['DRUG', 'CHEMICAL', 'MEDICATION']:
                    drug_name = entity['word']
                    confidence = entity['score']
                    
                    # Extract additional information around the drug mention
                    drug_context = self.extract_context_around_drug(text, drug_name)
                    
                    drug = ExtractedDrug(
                        name=drug_name,
                        confidence=confidence,
                        **drug_context
                    )
                    drugs.append(drug)
        
        except Exception as e:
            print(f"⚠️ HuggingFace extraction error: {e}")
        
        return drugs
    
    def extract_with_rules(self, text: str) -> List[ExtractedDrug]:
        """Extract drugs using rule-based patterns"""
        drugs = []
        
        # Find potential drug names
        drug_names = self.find_drug_names(text)
        
        for drug_name in drug_names:
            # Extract context around each drug
            drug_context = self.extract_context_around_drug(text, drug_name)
            
            drug = ExtractedDrug(
                name=drug_name,
                confidence=0.7,  # Rule-based confidence
                **drug_context
            )
            drugs.append(drug)
        
        return drugs
    
    def find_drug_names(self, text: str) -> List[str]:
        """Find potential drug names using patterns"""
        drug_names = set()
        
        # Common drug name patterns
        patterns = [
            r'\b([A-Z][a-z]+(?:in|ol|ide|ine|ate|ium|ic|an|ex))\b',
            r'\b([A-Z][a-z]*[A-Z][a-z]*)\b',  # CamelCase
            r'\b([A-Z]{2,})\b'  # All caps (abbreviations)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 2 and not match.lower() in ['the', 'and', 'for', 'with', 'take', 'give']:
                    drug_names.add(match)
        
        return list(drug_names)
    
    def extract_context_around_drug(self, text: str, drug_name: str) -> Dict[str, Any]:
        """Extract dosage, frequency, route, etc. around a drug mention"""
        context = {}
        
        # Find the drug mention and get surrounding context
        drug_pattern = re.compile(rf'\b{re.escape(drug_name)}\b', re.IGNORECASE)
        match = drug_pattern.search(text)
        
        if match:
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context_text = text[start:end]
            
            # Extract dosage
            dosage_info = self.extract_dosage(context_text)
            if dosage_info:
                context.update(dosage_info)
            
            # Extract frequency
            frequency = self.extract_frequency(context_text)
            if frequency:
                context['frequency'] = frequency
            
            # Extract route
            route = self.extract_route(context_text)
            if route:
                context['route'] = route
            
            # Extract duration
            duration = self.extract_duration(context_text)
            if duration:
                context['duration'] = duration
        
        return context
    
    def extract_dosage(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract dosage information"""
        for pattern in self.dosage_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    try:
                        dosage = float(groups[0])
                        unit = groups[-1].lower()
                        return {'dosage': dosage, 'unit': unit}
                    except ValueError:
                        continue
        return None
    
    def extract_frequency(self, text: str) -> Optional[str]:
        """Extract frequency information"""
        for pattern in self.frequency_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).lower()
        return None
    
    def extract_route(self, text: str) -> Optional[str]:
        """Extract route of administration"""
        for pattern in self.route_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                route_map = {
                    'oral': 'oral', 'orally': 'oral', 'by mouth': 'oral', 'po': 'oral',
                    'intravenous': 'IV', 'iv': 'IV', 'i.v.': 'IV',
                    'intramuscular': 'IM', 'im': 'IM', 'i.m.': 'IM',
                    'subcutaneous': 'SC', 'sc': 'SC', 'sq': 'SC', 's.c.': 'SC',
                    'topical': 'topical', 'topically': 'topical', 'applied': 'topical',
                    'inhaled': 'inhaled', 'inhalation': 'inhaled', 'nebulized': 'inhaled'
                }
                matched_text = match.group(0).lower()
                return route_map.get(matched_text, matched_text)
        return None
    
    def extract_duration(self, text: str) -> Optional[str]:
        """Extract treatment duration"""
        for pattern in self.duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None
    
    def merge_drug_extractions(self, drugs: List[ExtractedDrug]) -> List[ExtractedDrug]:
        """Merge duplicate drug extractions and combine information"""
        merged = {}
        
        for drug in drugs:
            key = drug.name.lower()
            
            if key in merged:
                # Merge information, keeping the one with higher confidence
                existing = merged[key]
                if drug.confidence > existing.confidence:
                    # Update with new drug but keep existing non-None values
                    for field in ['dosage', 'unit', 'frequency', 'route', 'duration', 'instructions']:
                        if getattr(drug, field) is None and getattr(existing, field) is not None:
                            setattr(drug, field, getattr(existing, field))
                    merged[key] = drug
                else:
                    # Keep existing but update None fields
                    for field in ['dosage', 'unit', 'frequency', 'route', 'duration', 'instructions']:
                        if getattr(existing, field) is None and getattr(drug, field) is not None:
                            setattr(existing, field, getattr(drug, field))
            else:
                merged[key] = drug
        
        return list(merged.values())
    
    def extract_from_prescription(self, prescription_text: str) -> List[ExtractedDrug]:
        """Extract drugs from prescription text with enhanced accuracy"""
        # Preprocess prescription text
        text = self.preprocess_prescription_text(prescription_text)
        
        # Extract drugs
        drugs = self.extract_drugs_from_text(text)
        
        # Post-process and validate
        validated_drugs = self.validate_extractions(drugs)
        
        return validated_drugs
    
    def preprocess_prescription_text(self, text: str) -> str:
        """Preprocess prescription-specific text"""
        # Common prescription format patterns
        text = re.sub(r'Rx:', '', text)
        text = re.sub(r'Sig:', 'Instructions:', text)
        text = re.sub(r'Disp:', 'Dispense:', text)
        
        return self.preprocess_text(text)
    
    def validate_extractions(self, drugs: List[ExtractedDrug]) -> List[ExtractedDrug]:
        """Validate and clean extracted drug information"""
        validated = []
        
        for drug in drugs:
            # Basic validation
            if len(drug.name) < 2:
                continue
            
            # Validate dosage units
            if drug.unit and drug.unit.lower() not in [u.lower() for u in DOSAGE_UNITS]:
                drug.unit = None
            
            # Validate frequency
            if drug.frequency:
                drug.frequency = self.normalize_frequency(drug.frequency)
            
            validated.append(drug)
        
        return validated
    
    def normalize_frequency(self, frequency: str) -> str:
        """Normalize frequency to standard format"""
        freq_map = {
            'once daily': 'once daily',
            'twice daily': 'twice daily',
            'three times daily': 'three times daily',
            'four times daily': 'four times daily',
            'qd': 'once daily',
            'bid': 'twice daily',
            'tid': 'three times daily',
            'qid': 'four times daily',
            'prn': 'as needed',
            'ac': 'before meals',
            'pc': 'after meals',
            'hs': 'at bedtime'
        }
        
        freq_lower = frequency.lower().strip()
        return freq_map.get(freq_lower, frequency)

# Example usage and testing
if __name__ == "__main__":
    extractor = DrugNLPExtractor()
    
    # Test with sample prescription text
    sample_texts = [
        "Take Metformin 500mg twice daily with meals for diabetes management.",
        "Prescribe Lisinopril 10mg once daily for hypertension. Patient should take Aspirin 81mg daily for cardioprotection.",
        "Rx: Amoxicillin 875mg PO BID x 10 days for bacterial infection. Sig: Take with food to reduce GI upset."
    ]
    
    print("🧪 Testing NLP Drug Extraction:")
    print("=" * 50)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nTest {i}: {text}")
        drugs = extractor.extract_drugs_from_text(text)
        
        for drug in drugs:
            print(f"  Drug: {drug.name}")
            print(f"  Dosage: {drug.dosage} {drug.unit}")
            print(f"  Frequency: {drug.frequency}")
            print(f"  Route: {drug.route}")
            print(f"  Confidence: {drug.confidence:.2f}")
            print("  ---")
