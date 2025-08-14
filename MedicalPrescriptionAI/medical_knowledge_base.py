# Medical Knowledge Base for RAG System
import os
import json
from typing import List, Dict, Any
import pickle
import numpy as np

# Fallback imports - use simpler alternatives if heavy packages not available
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ ChromaDB not available, using simple file-based storage")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ Sentence Transformers not available, using simple text matching")

class MedicalKnowledgeBase:
    def __init__(self, db_path: str = "./medical_knowledge_db"):
        self.db_path = db_path
        self.knowledge_data = []

        if CHROMADB_AVAILABLE:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name="medical_knowledge",
                metadata={"description": "Medical knowledge for RAG system"}
            )
        else:
            # Use simple file-based storage
            os.makedirs(db_path, exist_ok=True)
            self.data_file = os.path.join(db_path, "medical_data.json")
            self.load_simple_data()

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedding_model = None

    def load_simple_data(self):
        """Load data from simple JSON file"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.knowledge_data = json.load(f)
        else:
            self.knowledge_data = []

    def save_simple_data(self):
        """Save data to simple JSON file"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_data, f, indent=2, ensure_ascii=False)

    def add_medical_knowledge(self, texts: List[str], sources: List[str], categories: List[str]):
        """Add medical knowledge to the vector database"""
        if CHROMADB_AVAILABLE and self.embedding_model:
            # Use ChromaDB with embeddings
            embeddings = self.embedding_model.encode(texts)

            # Create unique IDs
            ids = [f"med_{i}" for i in range(len(texts))]

            # Create metadata
            metadatas = [
                {"source": source, "category": category}
                for source, category in zip(sources, categories)
            ]

            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        else:
            # Use simple storage
            for text, source, category in zip(texts, sources, categories):
                self.knowledge_data.append({
                    'text': text,
                    'source': source,
                    'category': category
                })
            if not CHROMADB_AVAILABLE:
                self.save_simple_data()
        
    def search_medical_knowledge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant medical knowledge"""

        if CHROMADB_AVAILABLE and self.embedding_model:
            # Use ChromaDB with embeddings
            query_embedding = self.embedding_model.encode([query])

            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )

            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'source': results['metadatas'][0][i]['source'],
                    'category': results['metadatas'][0][i]['category'],
                    'distance': results['distances'][0][i]
                })

            return formatted_results
        else:
            # Use simple text matching
            return self.simple_text_search(query, top_k)

    def simple_text_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Simple text-based search when embeddings are not available"""
        query_lower = query.lower()
        scored_results = []

        for item in self.knowledge_data:
            text_lower = item['text'].lower()
            # Simple scoring based on keyword matches
            score = 0
            for word in query_lower.split():
                if word in text_lower:
                    score += 1

            if score > 0:
                scored_results.append({
                    'text': item['text'],
                    'source': item['source'],
                    'category': item['category'],
                    'distance': 1.0 - (score / len(query_lower.split()))  # Convert to distance
                })

        # Sort by score (lower distance = better match)
        scored_results.sort(key=lambda x: x['distance'])
        return scored_results[:top_k]
    
    def initialize_with_sample_data(self):
        """Initialize database with sample medical knowledge"""
        sample_medical_data = [
            # General Medicine
            {
                "text": "Hypertension (high blood pressure) is a condition where blood pressure in arteries is persistently elevated. Normal blood pressure is typically below 120/80 mmHg. Hypertension is diagnosed when readings consistently exceed 140/90 mmHg.",
                "source": "Medical Textbook - Cardiology",
                "category": "cardiovascular"
            },
            {
                "text": "Type 2 diabetes is a chronic condition affecting how the body processes blood sugar (glucose). It occurs when the body becomes resistant to insulin or doesn't produce enough insulin. Symptoms include increased thirst, frequent urination, and fatigue.",
                "source": "Medical Textbook - Endocrinology", 
                "category": "endocrine"
            },
            {
                "text": "Pneumonia is an infection that inflames air sacs in one or both lungs. Symptoms include cough with phlegm, fever, chills, and difficulty breathing. It can be caused by bacteria, viruses, or fungi.",
                "source": "Medical Textbook - Pulmonology",
                "category": "respiratory"
            },
            # Pharmacology
            {
                "text": "Acetaminophen (paracetamol) is a pain reliever and fever reducer. Adult dosage is typically 325-650mg every 4-6 hours, not exceeding 3000mg per day. Overdose can cause severe liver damage.",
                "source": "Pharmacology Reference",
                "category": "pharmacology"
            },
            {
                "text": "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for pain, fever, and inflammation. Typical adult dose is 200-400mg every 4-6 hours. Should be taken with food to reduce stomach irritation.",
                "source": "Pharmacology Reference",
                "category": "pharmacology"
            },
            # Symptoms
            {
                "text": "Chest pain can have many causes including heart attack, angina, muscle strain, or acid reflux. Seek immediate medical attention if chest pain is severe, crushing, or accompanied by shortness of breath, nausea, or sweating.",
                "source": "Emergency Medicine Guidelines",
                "category": "symptoms"
            },
            {
                "text": "Headaches can be tension-type, migraine, or cluster headaches. Red flag symptoms requiring immediate medical attention include sudden severe headache, headache with fever and neck stiffness, or headache after head injury.",
                "source": "Neurology Guidelines",
                "category": "symptoms"
            },
            # Treatments
            {
                "text": "First aid for minor cuts: Clean hands, stop bleeding with direct pressure, clean wound with water, apply antibiotic ointment if available, cover with sterile bandage. Seek medical care for deep cuts or if bleeding doesn't stop.",
                "source": "First Aid Manual",
                "category": "treatment"
            },
            {
                "text": "RICE protocol for sprains: Rest the injured area, Ice for 15-20 minutes every 2-3 hours for first 48 hours, Compression with elastic bandage, Elevation above heart level when possible.",
                "source": "Sports Medicine Guidelines",
                "category": "treatment"
            }
        ]
        
        texts = [item["text"] for item in sample_medical_data]
        sources = [item["source"] for item in sample_medical_data]
        categories = [item["category"] for item in sample_medical_data]
        
        self.add_medical_knowledge(texts, sources, categories)
        print(f"Initialized medical knowledge base with {len(sample_medical_data)} entries")
