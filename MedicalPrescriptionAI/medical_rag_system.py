# Medical RAG (Retrieval-Augmented Generation) System
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from medical_knowledge_base import MedicalKnowledgeBase
from medical_ai_config import *
import torch
from typing import List, Dict, Any

class MedicalRAGSystem:
    def __init__(self):
        self.knowledge_base = MedicalKnowledgeBase(VECTOR_DB_PATH)
        self.setup_models()
        
    def setup_models(self):
        """Initialize the medical language models"""
        try:
            # Use a medical-focused model for better medical understanding
            model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
            
            # For question answering, we'll use a general model that works well
            self.qa_pipeline = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                tokenizer="deepset/roberta-base-squad2"
            )
            
            # For text generation, use a medical-aware model
            self.text_generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                tokenizer="microsoft/DialoGPT-medium",
                max_length=512,
                do_sample=True,
                temperature=0.7
            )
            
            print("✅ Medical models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            # Fallback to simpler models
            self.qa_pipeline = None
            self.text_generator = None
    
    def retrieve_relevant_knowledge(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        """Retrieve relevant medical knowledge for the query"""
        return self.knowledge_base.search_medical_knowledge(query, top_k)
    
    def generate_medical_response(self, query: str) -> Dict[str, Any]:
        """Generate a comprehensive medical response using RAG"""
        
        # Step 1: Retrieve relevant knowledge
        relevant_docs = self.retrieve_relevant_knowledge(query)
        
        if not relevant_docs:
            return {
                "answer": "I couldn't find relevant medical information for your query. Please consult a healthcare professional.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Step 2: Combine retrieved knowledge
        context = self.combine_retrieved_knowledge(relevant_docs)
        
        # Step 3: Generate answer using the context
        answer = self.generate_answer_from_context(query, context, relevant_docs)
        
        # Step 4: Add medical disclaimer
        final_answer = self.add_medical_disclaimer(answer)
        
        return {
            "answer": final_answer,
            "sources": [doc["source"] for doc in relevant_docs[:3]],
            "confidence": self.calculate_confidence(relevant_docs),
            "retrieved_docs": relevant_docs
        }
    
    def combine_retrieved_knowledge(self, docs: List[Dict[str, Any]]) -> str:
        """Combine retrieved documents into a coherent context"""
        context_parts = []
        for i, doc in enumerate(docs[:3]):  # Use top 3 most relevant
            context_parts.append(f"Source {i+1}: {doc['text']}")
        
        return "\n\n".join(context_parts)
    
    def generate_answer_from_context(self, query: str, context: str, docs: List[Dict[str, Any]]) -> str:
        """Generate answer using retrieved context"""
        
        # Try using QA pipeline if available
        if self.qa_pipeline:
            try:
                result = self.qa_pipeline(question=query, context=context)
                if result['score'] > 0.3:  # Confidence threshold
                    return result['answer']
            except Exception as e:
                print(f"QA pipeline error: {e}")
        
        # Fallback: Use the most relevant document
        if docs:
            best_doc = docs[0]
            return f"Based on medical knowledge: {best_doc['text']}"
        
        return "I found some relevant information, but I recommend consulting a healthcare professional for personalized advice."
    
    def calculate_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval results"""
        if not docs:
            return 0.0
        
        # Use distance/similarity scores to calculate confidence
        avg_distance = sum(doc.get('distance', 1.0) for doc in docs) / len(docs)
        confidence = max(0.0, 1.0 - avg_distance)  # Convert distance to confidence
        return min(confidence, 0.95)  # Cap at 95% to maintain humility
    
    def add_medical_disclaimer(self, answer: str) -> str:
        """Add appropriate medical disclaimers to the response"""
        
        # Check if answer contains medical advice indicators
        medical_keywords = ['take', 'dose', 'medication', 'treatment', 'diagnosis', 'prescribe']
        contains_medical_advice = any(keyword in answer.lower() for keyword in medical_keywords)
        
        if contains_medical_advice:
            disclaimer = "\n\n⚠️ **Important**: This information is for educational purposes only. Always consult with a qualified healthcare professional before making any medical decisions."
        else:
            disclaimer = "\n\n💡 **Note**: For personalized medical advice, please consult with a healthcare professional."
        
        return answer + disclaimer
    
    def initialize_knowledge_base(self):
        """Initialize the knowledge base with sample medical data"""
        self.knowledge_base.initialize_with_sample_data()
        print("✅ Medical knowledge base initialized")
    
    def add_custom_medical_knowledge(self, texts: List[str], sources: List[str], categories: List[str]):
        """Add custom medical knowledge to the system"""
        self.knowledge_base.add_medical_knowledge(texts, sources, categories)
        print(f"✅ Added {len(texts)} new medical knowledge entries")

# Example usage and testing
if __name__ == "__main__":
    # Initialize the medical RAG system
    medical_ai = MedicalRAGSystem()
    medical_ai.initialize_knowledge_base()
    
    # Test queries
    test_queries = [
        "What is hypertension?",
        "What are the symptoms of diabetes?",
        "How should I treat a sprain?",
        "What is the dosage for ibuprofen?",
        "What should I do for chest pain?"
    ]
    
    print("\n🏥 Testing Medical RAG System:\n")
    
    for query in test_queries:
        print(f"❓ Query: {query}")
        response = medical_ai.generate_medical_response(query)
        print(f"🤖 Answer: {response['answer']}")
        print(f"📚 Sources: {', '.join(response['sources'])}")
        print(f"🎯 Confidence: {response['confidence']:.2f}")
        print("-" * 80)
