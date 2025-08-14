# Age-Specific Dosage Recommendation System
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

class AgeGroup(Enum):
    NEONATE = "neonate"          # 0-1 month
    INFANT = "infant"            # 1 month - 2 years
    CHILD = "child"              # 2-12 years
    ADOLESCENT = "adolescent"    # 12-18 years
    ADULT = "adult"              # 18-65 years
    ELDERLY = "elderly"          # 65+ years

@dataclass
class PatientProfile:
    age: float  # in years
    weight: Optional[float] = None  # in kg
    height: Optional[float] = None  # in cm
    gender: Optional[str] = None
    kidney_function: str = "normal"  # normal, mild, moderate, severe
    liver_function: str = "normal"   # normal, mild, moderate, severe
    pregnancy_status: bool = False

@dataclass
class DosageRecommendation:
    drug_name: str
    recommended_dose: float
    unit: str
    frequency: str
    route: str
    max_daily_dose: float
    age_group: AgeGroup
    weight_based: bool
    adjustments: List[str]
    warnings: List[str]
    contraindications: List[str]

@dataclass
class DrugDosageInfo:
    drug_name: str
    adult_dose: float
    unit: str
    frequency: str
    route: str
    max_daily_dose: float
    weight_based_dosing: bool
    pediatric_dosing: Dict[str, Any]
    renal_adjustment: Dict[str, float]
    hepatic_adjustment: Dict[str, float]
    pregnancy_category: str
    contraindications: List[str]

class DosageCalculator:
    def __init__(self):
        self.drug_dosage_database = self.load_dosage_database()
        self.age_group_ranges = {
            AgeGroup.NEONATE: (0, 0.08),      # 0-1 month
            AgeGroup.INFANT: (0.08, 2),       # 1 month - 2 years
            AgeGroup.CHILD: (2, 12),          # 2-12 years
            AgeGroup.ADOLESCENT: (12, 18),    # 12-18 years
            AgeGroup.ADULT: (18, 65),         # 18-65 years
            AgeGroup.ELDERLY: (65, 120)       # 65+ years
        }
    
    def load_dosage_database(self) -> Dict[str, DrugDosageInfo]:
        """Load comprehensive drug dosage database"""
        
        dosage_db = {
            "acetaminophen": DrugDosageInfo(
                drug_name="Acetaminophen",
                adult_dose=650,
                unit="mg",
                frequency="every 4-6 hours",
                route="oral",
                max_daily_dose=3000,
                weight_based_dosing=True,
                pediatric_dosing={
                    "neonate": {"dose_per_kg": 10, "max_dose": 40, "frequency": "every 6-8 hours"},
                    "infant": {"dose_per_kg": 15, "max_dose": 80, "frequency": "every 4-6 hours"},
                    "child": {"dose_per_kg": 15, "max_dose": 650, "frequency": "every 4-6 hours"},
                    "adolescent": {"dose_per_kg": 15, "max_dose": 650, "frequency": "every 4-6 hours"}
                },
                renal_adjustment={"mild": 1.0, "moderate": 0.75, "severe": 0.5},
                hepatic_adjustment={"mild": 0.75, "moderate": 0.5, "severe": 0.25},
                pregnancy_category="B",
                contraindications=["Severe liver disease"]
            ),
            
            "ibuprofen": DrugDosageInfo(
                drug_name="Ibuprofen",
                adult_dose=400,
                unit="mg",
                frequency="every 6-8 hours",
                route="oral",
                max_daily_dose=1200,
                weight_based_dosing=True,
                pediatric_dosing={
                    "neonate": {"dose_per_kg": 0, "max_dose": 0, "frequency": "contraindicated"},
                    "infant": {"dose_per_kg": 10, "max_dose": 200, "frequency": "every 6-8 hours"},
                    "child": {"dose_per_kg": 10, "max_dose": 400, "frequency": "every 6-8 hours"},
                    "adolescent": {"dose_per_kg": 10, "max_dose": 400, "frequency": "every 6-8 hours"}
                },
                renal_adjustment={"mild": 0.75, "moderate": 0.5, "severe": 0},
                hepatic_adjustment={"mild": 0.75, "moderate": 0.5, "severe": 0},
                pregnancy_category="C",
                contraindications=["Severe kidney disease", "Active GI bleeding", "Third trimester pregnancy"]
            ),
            
            "amoxicillin": DrugDosageInfo(
                drug_name="Amoxicillin",
                adult_dose=500,
                unit="mg",
                frequency="every 8 hours",
                route="oral",
                max_daily_dose=3000,
                weight_based_dosing=True,
                pediatric_dosing={
                    "neonate": {"dose_per_kg": 30, "max_dose": 125, "frequency": "every 12 hours"},
                    "infant": {"dose_per_kg": 40, "max_dose": 250, "frequency": "every 8 hours"},
                    "child": {"dose_per_kg": 40, "max_dose": 500, "frequency": "every 8 hours"},
                    "adolescent": {"dose_per_kg": 25, "max_dose": 500, "frequency": "every 8 hours"}
                },
                renal_adjustment={"mild": 1.0, "moderate": 0.75, "severe": 0.5},
                hepatic_adjustment={"mild": 1.0, "moderate": 1.0, "severe": 1.0},
                pregnancy_category="B",
                contraindications=["Penicillin allergy"]
            ),
            
            "metformin": DrugDosageInfo(
                drug_name="Metformin",
                adult_dose=500,
                unit="mg",
                frequency="twice daily",
                route="oral",
                max_daily_dose=2000,
                weight_based_dosing=False,
                pediatric_dosing={
                    "neonate": {"dose_per_kg": 0, "max_dose": 0, "frequency": "not recommended"},
                    "infant": {"dose_per_kg": 0, "max_dose": 0, "frequency": "not recommended"},
                    "child": {"dose_per_kg": 0, "max_dose": 500, "frequency": "twice daily"},
                    "adolescent": {"dose_per_kg": 0, "max_dose": 500, "frequency": "twice daily"}
                },
                renal_adjustment={"mild": 1.0, "moderate": 0.5, "severe": 0},
                hepatic_adjustment={"mild": 1.0, "moderate": 0.75, "severe": 0},
                pregnancy_category="B",
                contraindications=["Severe kidney disease", "Metabolic acidosis", "Severe liver disease"]
            ),
            
            "lisinopril": DrugDosageInfo(
                drug_name="Lisinopril",
                adult_dose=10,
                unit="mg",
                frequency="once daily",
                route="oral",
                max_daily_dose=40,
                weight_based_dosing=False,
                pediatric_dosing={
                    "neonate": {"dose_per_kg": 0, "max_dose": 0, "frequency": "contraindicated"},
                    "infant": {"dose_per_kg": 0, "max_dose": 0, "frequency": "contraindicated"},
                    "child": {"dose_per_kg": 0.1, "max_dose": 5, "frequency": "once daily"},
                    "adolescent": {"dose_per_kg": 0.1, "max_dose": 10, "frequency": "once daily"}
                },
                renal_adjustment={"mild": 0.75, "moderate": 0.5, "severe": 0.25},
                hepatic_adjustment={"mild": 1.0, "moderate": 1.0, "severe": 1.0},
                pregnancy_category="D",
                contraindications=["Pregnancy", "Angioedema history", "Bilateral renal artery stenosis"]
            )
        }
        
        return dosage_db
    
    def determine_age_group(self, age: float) -> AgeGroup:
        """Determine age group based on patient age"""
        for age_group, (min_age, max_age) in self.age_group_ranges.items():
            if min_age <= age < max_age:
                return age_group
        return AgeGroup.ELDERLY  # Default for very old patients
    
    def calculate_dosage(self, drug_name: str, patient: PatientProfile) -> DosageRecommendation:
        """Calculate age-appropriate dosage for a specific drug"""
        
        # Normalize drug name
        drug_key = drug_name.lower().replace(" ", "").replace("-", "")
        
        if drug_key not in self.drug_dosage_database:
            raise ValueError(f"Drug '{drug_name}' not found in dosage database")
        
        drug_info = self.drug_dosage_database[drug_key]
        age_group = self.determine_age_group(patient.age)
        
        # Check contraindications
        contraindications = self.check_contraindications(drug_info, patient, age_group)
        if contraindications:
            return DosageRecommendation(
                drug_name=drug_info.drug_name,
                recommended_dose=0,
                unit=drug_info.unit,
                frequency="CONTRAINDICATED",
                route=drug_info.route,
                max_daily_dose=0,
                age_group=age_group,
                weight_based=False,
                adjustments=[],
                warnings=[],
                contraindications=contraindications
            )
        
        # Calculate base dose
        base_dose = self.calculate_base_dose(drug_info, patient, age_group)
        
        # Apply adjustments
        adjusted_dose, adjustments = self.apply_dose_adjustments(
            base_dose, drug_info, patient, age_group
        )
        
        # Generate warnings
        warnings = self.generate_warnings(drug_info, patient, age_group)
        
        # Calculate max daily dose
        max_daily = self.calculate_max_daily_dose(drug_info, patient, age_group)
        
        return DosageRecommendation(
            drug_name=drug_info.drug_name,
            recommended_dose=adjusted_dose,
            unit=drug_info.unit,
            frequency=drug_info.frequency,
            route=drug_info.route,
            max_daily_dose=max_daily,
            age_group=age_group,
            weight_based=drug_info.weight_based_dosing and patient.weight is not None,
            adjustments=adjustments,
            warnings=warnings,
            contraindications=[]
        )
    
    def calculate_base_dose(self, drug_info: DrugDosageInfo, patient: PatientProfile, age_group: AgeGroup) -> float:
        """Calculate base dose based on age group and weight"""
        
        if age_group == AgeGroup.ADULT:
            return drug_info.adult_dose
        
        # Pediatric dosing
        age_group_key = age_group.value
        if age_group_key in drug_info.pediatric_dosing:
            pediatric_info = drug_info.pediatric_dosing[age_group_key]
            
            if pediatric_info["dose_per_kg"] > 0 and patient.weight:
                # Weight-based dosing
                calculated_dose = pediatric_info["dose_per_kg"] * patient.weight
                return min(calculated_dose, pediatric_info["max_dose"])
            else:
                # Fixed dosing
                return pediatric_info["max_dose"]
        
        # Elderly dosing (typically reduced adult dose)
        if age_group == AgeGroup.ELDERLY:
            return drug_info.adult_dose * 0.75  # 25% reduction for elderly
        
        return drug_info.adult_dose
    
    def apply_dose_adjustments(self, base_dose: float, drug_info: DrugDosageInfo, 
                             patient: PatientProfile, age_group: AgeGroup) -> Tuple[float, List[str]]:
        """Apply dose adjustments for organ function"""
        
        adjusted_dose = base_dose
        adjustments = []
        
        # Renal adjustment
        if patient.kidney_function != "normal":
            renal_factor = drug_info.renal_adjustment.get(patient.kidney_function, 1.0)
            if renal_factor < 1.0:
                adjusted_dose *= renal_factor
                adjustments.append(f"Dose reduced by {int((1-renal_factor)*100)}% for {patient.kidney_function} kidney function")
            elif renal_factor == 0:
                adjusted_dose = 0
                adjustments.append("Drug contraindicated in severe kidney disease")
        
        # Hepatic adjustment
        if patient.liver_function != "normal":
            hepatic_factor = drug_info.hepatic_adjustment.get(patient.liver_function, 1.0)
            if hepatic_factor < 1.0:
                adjusted_dose *= hepatic_factor
                adjustments.append(f"Dose reduced by {int((1-hepatic_factor)*100)}% for {patient.liver_function} liver function")
            elif hepatic_factor == 0:
                adjusted_dose = 0
                adjustments.append("Drug contraindicated in severe liver disease")
        
        # Age-specific adjustments
        if age_group == AgeGroup.ELDERLY:
            adjustments.append("Elderly patient - monitor for increased sensitivity")
        elif age_group in [AgeGroup.NEONATE, AgeGroup.INFANT]:
            adjustments.append("Pediatric patient - close monitoring required")
        
        return round(adjusted_dose, 1), adjustments
    
    def check_contraindications(self, drug_info: DrugDosageInfo, patient: PatientProfile, 
                               age_group: AgeGroup) -> List[str]:
        """Check for contraindications"""
        contraindications = []
        
        # Pregnancy contraindications
        if patient.pregnancy_status:
            if drug_info.pregnancy_category in ["D", "X"]:
                contraindications.append(f"Pregnancy category {drug_info.pregnancy_category} - avoid in pregnancy")
        
        # Age-specific contraindications
        age_group_key = age_group.value
        if age_group_key in drug_info.pediatric_dosing:
            pediatric_info = drug_info.pediatric_dosing[age_group_key]
            if "contraindicated" in pediatric_info["frequency"].lower():
                contraindications.append(f"Contraindicated in {age_group.value} patients")
        
        # Organ function contraindications
        if patient.kidney_function == "severe":
            if drug_info.renal_adjustment.get("severe", 1.0) == 0:
                contraindications.append("Contraindicated in severe kidney disease")
        
        if patient.liver_function == "severe":
            if drug_info.hepatic_adjustment.get("severe", 1.0) == 0:
                contraindications.append("Contraindicated in severe liver disease")
        
        # Drug-specific contraindications
        contraindications.extend(drug_info.contraindications)
        
        return contraindications
    
    def generate_warnings(self, drug_info: DrugDosageInfo, patient: PatientProfile, 
                         age_group: AgeGroup) -> List[str]:
        """Generate safety warnings"""
        warnings = []
        
        # Age-specific warnings
        if age_group == AgeGroup.ELDERLY:
            warnings.append("⚠️ Elderly patient - increased risk of adverse effects")
        elif age_group in [AgeGroup.NEONATE, AgeGroup.INFANT]:
            warnings.append("⚠️ Pediatric patient - requires careful monitoring")
        
        # Pregnancy warnings
        if patient.pregnancy_status:
            if drug_info.pregnancy_category == "C":
                warnings.append("⚠️ Pregnancy category C - use only if benefit outweighs risk")
            elif drug_info.pregnancy_category == "B":
                warnings.append("💡 Pregnancy category B - generally safe but monitor")
        
        # Organ function warnings
        if patient.kidney_function in ["moderate", "severe"]:
            warnings.append("⚠️ Monitor kidney function and adjust dose as needed")
        
        if patient.liver_function in ["moderate", "severe"]:
            warnings.append("⚠️ Monitor liver function and watch for toxicity")
        
        # Weight-based dosing warnings
        if drug_info.weight_based_dosing and not patient.weight:
            warnings.append("⚠️ Weight-based dosing recommended - patient weight needed")
        
        return warnings
    
    def calculate_max_daily_dose(self, drug_info: DrugDosageInfo, patient: PatientProfile, 
                                age_group: AgeGroup) -> float:
        """Calculate maximum daily dose"""
        
        if age_group == AgeGroup.ADULT:
            max_dose = drug_info.max_daily_dose
        else:
            # Pediatric max dose
            age_group_key = age_group.value
            if age_group_key in drug_info.pediatric_dosing:
                pediatric_info = drug_info.pediatric_dosing[age_group_key]
                if drug_info.weight_based_dosing and patient.weight:
                    max_dose = pediatric_info["dose_per_kg"] * patient.weight * 4  # Assuming max 4 doses/day
                else:
                    max_dose = pediatric_info["max_dose"] * 4
            else:
                max_dose = drug_info.max_daily_dose * 0.5  # Conservative for pediatric
        
        # Apply organ function adjustments
        if patient.kidney_function != "normal":
            renal_factor = drug_info.renal_adjustment.get(patient.kidney_function, 1.0)
            max_dose *= renal_factor
        
        if patient.liver_function != "normal":
            hepatic_factor = drug_info.hepatic_adjustment.get(patient.liver_function, 1.0)
            max_dose *= hepatic_factor
        
        return round(max_dose, 1)
    
    def get_dosage_recommendations(self, drug_list: List[str], patient: PatientProfile) -> List[DosageRecommendation]:
        """Get dosage recommendations for multiple drugs"""
        recommendations = []
        
        for drug_name in drug_list:
            try:
                recommendation = self.calculate_dosage(drug_name, patient)
                recommendations.append(recommendation)
            except ValueError as e:
                # Create a placeholder recommendation for unknown drugs
                recommendations.append(DosageRecommendation(
                    drug_name=drug_name,
                    recommended_dose=0,
                    unit="unknown",
                    frequency="unknown",
                    route="unknown",
                    max_daily_dose=0,
                    age_group=self.determine_age_group(patient.age),
                    weight_based=False,
                    adjustments=[],
                    warnings=[f"⚠️ Drug not found in database: {str(e)}"],
                    contraindications=[]
                ))
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    calculator = DosageCalculator()
    
    # Test patients
    test_patients = [
        PatientProfile(age=0.5, weight=4.5, gender="male"),  # 6-month infant
        PatientProfile(age=8, weight=25, gender="female"),   # 8-year-old child
        PatientProfile(age=35, weight=70, gender="male"),    # Adult
        PatientProfile(age=75, weight=65, kidney_function="moderate"),  # Elderly with kidney issues
        PatientProfile(age=28, weight=60, pregnancy_status=True)  # Pregnant woman
    ]
    
    test_drugs = ["acetaminophen", "ibuprofen", "amoxicillin", "metformin", "lisinopril"]
    
    print("🧪 Testing Dosage Calculator:")
    print("=" * 60)
    
    for i, patient in enumerate(test_patients, 1):
        print(f"\nPatient {i}: Age {patient.age} years, Weight {patient.weight}kg")
        if patient.pregnancy_status:
            print("  Status: Pregnant")
        if patient.kidney_function != "normal":
            print(f"  Kidney function: {patient.kidney_function}")
        
        print("  Dosage Recommendations:")
        
        for drug in test_drugs[:3]:  # Test first 3 drugs
            try:
                rec = calculator.calculate_dosage(drug, patient)
                print(f"    {rec.drug_name}: {rec.recommended_dose}{rec.unit} {rec.frequency}")
                if rec.contraindications:
                    print(f"      ❌ CONTRAINDICATED: {', '.join(rec.contraindications)}")
                if rec.adjustments:
                    print(f"      📝 Adjustments: {', '.join(rec.adjustments)}")
                if rec.warnings:
                    print(f"      ⚠️ Warnings: {', '.join(rec.warnings)}")
            except Exception as e:
                print(f"    {drug}: Error - {e}")
        
        print("-" * 40)
