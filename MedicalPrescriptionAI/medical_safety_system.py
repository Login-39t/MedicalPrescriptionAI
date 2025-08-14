# Medical Safety and Disclaimer System
import re
from typing import List, Dict, Tuple, Any
from medical_ai_config import MEDICAL_DISCLAIMER

class MedicalSafetySystem:
    def __init__(self):
        self.emergency_keywords = [
            'emergency', 'urgent', 'severe pain', 'chest pain', 'heart attack', 
            'stroke', 'bleeding', 'unconscious', 'difficulty breathing', 
            'allergic reaction', 'overdose', 'suicide', 'poisoning'
        ]
        
        self.prescription_keywords = [
            'prescribe', 'prescription', 'medication dosage', 'drug interaction',
            'how much to take', 'when to take', 'stop taking', 'increase dose'
        ]
        
        self.diagnosis_keywords = [
            'do i have', 'am i sick', 'what disease', 'diagnose me', 
            'what condition', 'is this cancer', 'do i need surgery'
        ]
    
    def analyze_query_risk(self, query: str) -> Dict[str, Any]:
        """Analyze the medical query for risk factors"""
        query_lower = query.lower()
        
        risk_analysis = {
            'risk_level': 'low',
            'emergency_detected': False,
            'prescription_request': False,
            'diagnosis_request': False,
            'warnings': [],
            'required_disclaimers': []
        }
        
        # Check for emergency keywords
        for keyword in self.emergency_keywords:
            if keyword in query_lower:
                risk_analysis['emergency_detected'] = True
                risk_analysis['risk_level'] = 'emergency'
                risk_analysis['warnings'].append(f"Emergency keyword detected: '{keyword}'")
                break
        
        # Check for prescription requests
        for keyword in self.prescription_keywords:
            if keyword in query_lower:
                risk_analysis['prescription_request'] = True
                risk_analysis['risk_level'] = 'high'
                risk_analysis['warnings'].append("Prescription-related query detected")
                break
        
        # Check for diagnosis requests
        for keyword in self.diagnosis_keywords:
            if keyword in query_lower:
                risk_analysis['diagnosis_request'] = True
                if risk_analysis['risk_level'] == 'low':
                    risk_analysis['risk_level'] = 'medium'
                risk_analysis['warnings'].append("Diagnosis-related query detected")
                break
        
        return risk_analysis
    
    def generate_safety_response(self, risk_analysis: Dict[str, Any]) -> str:
        """Generate appropriate safety response based on risk analysis"""
        
        if risk_analysis['emergency_detected']:
            return self.get_emergency_response()
        
        elif risk_analysis['prescription_request']:
            return self.get_prescription_warning()
        
        elif risk_analysis['diagnosis_request']:
            return self.get_diagnosis_warning()
        
        else:
            return self.get_general_disclaimer()
    
    def get_emergency_response(self) -> str:
        """Response for emergency situations"""
        return """
🚨 **MEDICAL EMERGENCY DETECTED** 🚨

If you are experiencing a medical emergency:
• **Call emergency services immediately** (911 in US, 999 in UK, 112 in EU)
• **Go to the nearest emergency room**
• **Contact your local emergency medical services**

**Do not rely on AI for emergency medical situations.**

This AI cannot provide emergency medical care or replace immediate professional medical attention.
"""
    
    def get_prescription_warning(self) -> str:
        """Warning for prescription-related queries"""
        return """
💊 **PRESCRIPTION MEDICATION WARNING** 💊

**I cannot provide prescription medication advice.**

For medication-related questions:
• **Consult your prescribing physician**
• **Contact your pharmacist**
• **Call your healthcare provider**
• **Read medication labels and patient information**

**Never start, stop, or change medications without professional medical supervision.**
"""
    
    def get_diagnosis_warning(self) -> str:
        """Warning for diagnosis-related queries"""
        return """
🩺 **MEDICAL DIAGNOSIS WARNING** 🩺

**I cannot provide medical diagnoses.**

For health concerns:
• **Schedule an appointment with a healthcare provider**
• **Visit an urgent care center if needed**
• **Contact your primary care physician**
• **Seek professional medical evaluation**

**Only qualified healthcare professionals can provide accurate medical diagnoses.**
"""
    
    def get_general_disclaimer(self) -> str:
        """General medical disclaimer"""
        return """
ℹ️ **Medical Information Disclaimer** ℹ️

This information is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.

**Always consult with qualified healthcare professionals for medical decisions.**
"""
    
    def filter_response(self, response: str, risk_analysis: Dict[str, Any]) -> str:
        """Filter and modify response based on safety analysis"""
        
        # Remove potentially harmful advice
        harmful_phrases = [
            'you should take', 'i recommend taking', 'the dosage is',
            'you have', 'you are diagnosed with', 'this means you have'
        ]
        
        filtered_response = response
        for phrase in harmful_phrases:
            if phrase in response.lower():
                filtered_response = filtered_response.replace(phrase, 'healthcare professionals may recommend')
        
        # Add appropriate safety warnings
        safety_response = self.generate_safety_response(risk_analysis)
        
        return f"{filtered_response}\n\n{safety_response}"
    
    def validate_medical_response(self, query: str, response: str) -> Dict[str, Any]:
        """Validate the complete medical response for safety"""
        
        risk_analysis = self.analyze_query_risk(query)
        filtered_response = self.filter_response(response, risk_analysis)
        
        validation_result = {
            'original_response': response,
            'filtered_response': filtered_response,
            'risk_analysis': risk_analysis,
            'safety_score': self.calculate_safety_score(risk_analysis),
            'approved': True
        }
        
        # Block response if too risky
        if risk_analysis['risk_level'] == 'emergency':
            validation_result['approved'] = False
            validation_result['filtered_response'] = self.get_emergency_response()
        
        return validation_result
    
    def calculate_safety_score(self, risk_analysis: Dict[str, Any]) -> float:
        """Calculate safety score (0-1, where 1 is safest)"""
        
        if risk_analysis['emergency_detected']:
            return 0.1
        elif risk_analysis['prescription_request']:
            return 0.3
        elif risk_analysis['diagnosis_request']:
            return 0.5
        else:
            return 0.9
    
    def get_medical_disclaimer_banner(self) -> str:
        """Get the main medical disclaimer banner"""
        return MEDICAL_DISCLAIMER

# Example usage and testing
if __name__ == "__main__":
    safety_system = MedicalSafetySystem()
    
    test_queries = [
        "I'm having severe chest pain and can't breathe",  # Emergency
        "What dosage of ibuprofen should I take?",         # Prescription
        "Do I have cancer based on these symptoms?",       # Diagnosis
        "What is diabetes?",                               # General
        "How to treat a minor cut?"                        # General
    ]
    
    print("🛡️ Testing Medical Safety System:\n")
    
    for query in test_queries:
        print(f"❓ Query: {query}")
        risk_analysis = safety_system.analyze_query_risk(query)
        print(f"⚠️ Risk Level: {risk_analysis['risk_level']}")
        print(f"🚨 Warnings: {risk_analysis['warnings']}")
        
        safety_response = safety_system.generate_safety_response(risk_analysis)
        print(f"🛡️ Safety Response: {safety_response}")
        print("-" * 80)
