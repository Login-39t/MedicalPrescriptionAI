# Simple Flask Drug Analysis System (No FastAPI Dependencies)
from flask import Flask, render_template, request, jsonify
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'simple_drug_analysis_2024'

# Data Models
class InteractionSeverity(str, Enum):
    CONTRAINDICATED = "contraindicated"
    MAJOR = "major"
    MODERATE = "moderate"
    MINOR = "minor"
    UNKNOWN = "unknown"

class AgeGroup(str, Enum):
    NEONATE = "neonate"
    INFANT = "infant"
    CHILD = "child"
    ADOLESCENT = "adolescent"
    ADULT = "adult"
    ELDERLY = "elderly"

# Simple Drug Analysis System
class SimpleDrugAnalysisSystem:
    def __init__(self):
        self.setup_databases()
    
    def setup_databases(self):
        """Setup drug databases"""
        # Drug interaction database
        self.interactions = {
            ("warfarin", "aspirin"): {
                "severity": InteractionSeverity.MAJOR,
                "description": "Increased risk of bleeding due to additive anticoagulant effects",
                "mechanism": "Both drugs affect blood coagulation through different pathways",
                "clinical_effect": "Increased bleeding risk, prolonged PT/INR, potential hemorrhage",
                "management": "Monitor INR closely, consider dose reduction, watch for bleeding signs",
                "risk_score": 8,
                "evidence_level": "A"
            },
            ("metformin", "contrast_dye"): {
                "severity": InteractionSeverity.CONTRAINDICATED,
                "description": "Risk of lactic acidosis due to impaired renal clearance",
                "mechanism": "Contrast agents can cause acute kidney injury, reducing metformin clearance",
                "clinical_effect": "Lactic acidosis, kidney damage, metabolic complications",
                "management": "Discontinue metformin 48h before contrast, resume after kidney function confirmed",
                "risk_score": 10,
                "evidence_level": "A"
            },
            ("lisinopril", "potassium"): {
                "severity": InteractionSeverity.MODERATE,
                "description": "Risk of hyperkalemia",
                "mechanism": "ACE inhibitor reduces potassium excretion by kidneys",
                "clinical_effect": "Elevated serum potassium, cardiac arrhythmias",
                "management": "Monitor potassium levels regularly, adjust doses as needed",
                "risk_score": 6,
                "evidence_level": "B"
            }
        }
        
        # Dosage database
        self.dosages = {
            "metformin": {
                AgeGroup.CHILD: {"dose": 500, "unit": "mg", "frequency": "twice daily", "route": "oral"},
                AgeGroup.ADOLESCENT: {"dose": 850, "unit": "mg", "frequency": "twice daily", "route": "oral"},
                AgeGroup.ADULT: {"dose": 1000, "unit": "mg", "frequency": "twice daily", "route": "oral"},
                AgeGroup.ELDERLY: {"dose": 500, "unit": "mg", "frequency": "twice daily", "route": "oral"},
                "warnings": ["Monitor kidney function", "Risk of lactic acidosis"],
                "contraindications": ["Severe kidney disease", "Acute heart failure"]
            },
            "lisinopril": {
                AgeGroup.CHILD: {"dose": 2.5, "unit": "mg", "frequency": "once daily", "route": "oral"},
                AgeGroup.ADOLESCENT: {"dose": 5, "unit": "mg", "frequency": "once daily", "route": "oral"},
                AgeGroup.ADULT: {"dose": 10, "unit": "mg", "frequency": "once daily", "route": "oral"},
                AgeGroup.ELDERLY: {"dose": 5, "unit": "mg", "frequency": "once daily", "route": "oral"},
                "warnings": ["Monitor blood pressure", "Check kidney function"],
                "contraindications": ["Pregnancy", "Angioedema history"]
            },
            "warfarin": {
                AgeGroup.ADULT: {"dose": 5, "unit": "mg", "frequency": "once daily", "route": "oral"},
                AgeGroup.ELDERLY: {"dose": 2.5, "unit": "mg", "frequency": "once daily", "route": "oral"},
                "warnings": ["Monitor INR regularly", "Bleeding risk"],
                "contraindications": ["Active bleeding", "Pregnancy"]
            }
        }
        
        # Alternative medications database
        self.alternatives = {
            "warfarin": [
                {"name": "rivaroxaban", "reason": "Newer anticoagulant", "suitability": 85},
                {"name": "apixaban", "reason": "Lower bleeding risk", "suitability": 90},
                {"name": "dabigatran", "reason": "Reversible anticoagulant", "suitability": 80}
            ],
            "aspirin": [
                {"name": "clopidogrel", "reason": "Alternative antiplatelet", "suitability": 85},
                {"name": "acetaminophen", "reason": "Pain relief without bleeding risk", "suitability": 70}
            ],
            "metformin": [
                {"name": "glipizide", "reason": "Alternative diabetes medication", "suitability": 80},
                {"name": "sitagliptin", "reason": "DPP-4 inhibitor", "suitability": 85}
            ]
        }
    
    def get_age_group(self, age: float) -> AgeGroup:
        """Determine age group from age"""
        if age < 1/12:
            return AgeGroup.NEONATE
        elif age < 2:
            return AgeGroup.INFANT
        elif age < 12:
            return AgeGroup.CHILD
        elif age < 18:
            return AgeGroup.ADOLESCENT
        elif age < 65:
            return AgeGroup.ADULT
        else:
            return AgeGroup.ELDERLY
    
    def check_interaction(self, drug1: str, drug2: str) -> Optional[Dict]:
        """Check for drug interactions"""
        drug1_norm = drug1.lower().strip()
        drug2_norm = drug2.lower().strip()
        
        interaction = self.interactions.get((drug1_norm, drug2_norm)) or \
                     self.interactions.get((drug2_norm, drug1_norm))
        
        return interaction
    
    def calculate_dosage(self, drug_name: str, age: float, weight: Optional[float] = None, 
                        kidney_function: str = "normal", liver_function: str = "normal") -> Optional[Dict]:
        """Calculate appropriate dosage"""
        drug_name_norm = drug_name.lower().strip()
        
        if drug_name_norm not in self.dosages:
            return None
        
        drug_data = self.dosages[drug_name_norm]
        age_group = self.get_age_group(age)
        
        base_dosage = drug_data.get(age_group)
        if not base_dosage:
            base_dosage = drug_data.get(AgeGroup.ADULT)
        
        if not base_dosage:
            return None
        
        # Apply adjustments
        adjustments = {}
        dose_multiplier = 1.0
        
        if kidney_function == "mild":
            dose_multiplier *= 0.9
            adjustments["kidney"] = "Mild impairment: 10% dose reduction"
        elif kidney_function == "moderate":
            dose_multiplier *= 0.7
            adjustments["kidney"] = "Moderate impairment: 30% dose reduction"
        elif kidney_function == "severe":
            dose_multiplier *= 0.5
            adjustments["kidney"] = "Severe impairment: 50% dose reduction"
        
        if liver_function == "mild":
            dose_multiplier *= 0.9
            adjustments["liver"] = "Mild impairment: 10% dose reduction"
        elif liver_function == "moderate":
            dose_multiplier *= 0.7
            adjustments["liver"] = "Moderate impairment: 30% dose reduction"
        elif liver_function == "severe":
            dose_multiplier *= 0.5
            adjustments["liver"] = "Severe impairment: 50% dose reduction"
        
        if weight and weight < 50:
            dose_multiplier *= 0.8
            adjustments["weight"] = "Low weight: 20% dose reduction"
        elif weight and weight > 100:
            dose_multiplier *= 1.2
            adjustments["weight"] = "High weight: 20% dose increase"
        
        adjusted_dose = base_dosage["dose"] * dose_multiplier
        
        return {
            "drug_name": drug_name,
            "recommended_dose": round(adjusted_dose, 2),
            "unit": base_dosage["unit"],
            "frequency": base_dosage["frequency"],
            "route": base_dosage["route"],
            "age_group": age_group.value,
            "warnings": drug_data.get("warnings", []),
            "contraindications": drug_data.get("contraindications", []),
            "adjustments": adjustments
        }
    
    def get_alternatives(self, drug_name: str) -> List[Dict]:
        """Get alternative medications"""
        drug_name_norm = drug_name.lower().strip()
        return self.alternatives.get(drug_name_norm, [])
    
    def extract_drugs_from_text(self, text: str) -> List[Dict]:
        """Extract drug information from text using regex"""
        drugs = []
        
        # Common drug patterns
        drug_patterns = [
            r'(metformin|warfarin|lisinopril|aspirin|ibuprofen|acetaminophen)\s*(\d+(?:\.\d+)?)\s*(mg|g|ml)',
            r'(metformin|warfarin|lisinopril|aspirin|ibuprofen|acetaminophen)',
        ]
        
        for pattern in drug_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                drug_name = match.group(1)
                dosage = None
                unit = None
                
                if len(match.groups()) >= 3:
                    try:
                        dosage = float(match.group(2))
                        unit = match.group(3)
                    except (ValueError, IndexError):
                        pass
                
                # Extract frequency
                frequency = None
                freq_patterns = [
                    r'(once|twice|three times|four times)\s*(daily|a day|per day)',
                    r'(every \d+ hours?)',
                    r'(bid|tid|qid|qd)'
                ]
                
                for freq_pattern in freq_patterns:
                    freq_match = re.search(freq_pattern, text.lower())
                    if freq_match:
                        frequency = freq_match.group(0)
                        break
                
                drugs.append({
                    "name": drug_name,
                    "dosage": dosage,
                    "unit": unit,
                    "frequency": frequency,
                    "confidence": 0.8,
                    "extraction_method": "Regex"
                })
        
        return drugs

# Initialize the drug analysis system
drug_system = SimpleDrugAnalysisSystem()

# Flask Routes
@app.route('/')
def index():
    """Main drug analysis interface"""
    return render_template('drug_analysis.html')

@app.route('/api/check-interaction', methods=['POST'])
def check_interaction():
    """Check drug interaction"""
    try:
        data = request.get_json()
        drug1 = data.get('drug1', '')
        drug2 = data.get('drug2', '')
        
        if not drug1 or not drug2:
            return jsonify({"error": "Please enter both drug names"}), 400
        
        interaction = drug_system.check_interaction(drug1, drug2)
        
        if not interaction:
            return jsonify({
                "drug1": drug1,
                "drug2": drug2,
                "severity": InteractionSeverity.UNKNOWN.value,
                "description": "No known interaction found",
                "mechanism": "Unknown",
                "clinical_effect": "No known effects",
                "management": "Monitor for unexpected effects",
                "risk_score": 0,
                "evidence_level": "N/A"
            })
        
        return jsonify({
            "drug1": drug1,
            "drug2": drug2,
            "severity": interaction["severity"].value,
            "description": interaction["description"],
            "mechanism": interaction["mechanism"],
            "clinical_effect": interaction["clinical_effect"],
            "management": interaction["management"],
            "risk_score": interaction["risk_score"],
            "evidence_level": interaction["evidence_level"]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-dosage', methods=['POST'])
def get_dosage():
    """Get dosage recommendation"""
    try:
        data = request.get_json()
        drug_name = data.get('drug_name', '')
        patient_profile = data.get('patient_profile', {})
        
        age = patient_profile.get('age')
        weight = patient_profile.get('weight')
        kidney_function = patient_profile.get('kidney_function', 'normal')
        liver_function = patient_profile.get('liver_function', 'normal')
        
        if not drug_name or not age:
            return jsonify({"error": "Please enter drug name and patient age"}), 400
        
        dosage_info = drug_system.calculate_dosage(drug_name, age, weight, kidney_function, liver_function)
        
        if not dosage_info:
            return jsonify({"error": f"Dosage information not found for {drug_name}"}), 404
        
        return jsonify(dosage_info)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get-alternatives', methods=['POST'])
def get_alternatives():
    """Get alternative medications"""
    try:
        data = request.get_json()
        drug_name = data.get('drug_name', '')
        reason = data.get('reason', 'interaction')
        
        if not drug_name:
            return jsonify({"error": "Please enter a drug name"}), 400
        
        alternatives = drug_system.get_alternatives(drug_name)
        
        return jsonify({
            "original_drug": drug_name,
            "alternatives": alternatives,
            "reason": reason
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/extract-drugs', methods=['POST'])
def extract_drugs():
    """Extract drugs from text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "Please enter medical text"}), 400
        
        extracted_drugs = drug_system.extract_drugs_from_text(text)
        
        return jsonify(extracted_drugs)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/system-status')
def system_status():
    """Get system status"""
    return jsonify({
        "status": "operational",
        "version": "1.0.0",
        "features": [
            "Drug Interaction Detection",
            "Age-Specific Dosage Recommendations", 
            "Alternative Medication Suggestions",
            "Basic Drug Information Extraction"
        ],
        "supported_drugs": list(drug_system.dosages.keys()),
        "interaction_count": len(drug_system.interactions),
        "alternative_count": sum(len(alts) for alts in drug_system.alternatives.values()),
        "backend": "Simple Flask System"
    })

if __name__ == "__main__":
    print("🚀 Starting Simple Drug Analysis System...")
    print("📱 Access at: http://localhost:5003")
    print("💊 Features: Drug interactions, dosage calculator, alternatives, NLP extraction")
    app.run(debug=True, host='0.0.0.0', port=5003)
