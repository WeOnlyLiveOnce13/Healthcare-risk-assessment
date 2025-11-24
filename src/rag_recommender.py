import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL
from data_loader import load_guidelines_pdf, chunk_text


class RAGRecommender:
    """RAG system for guideline-based recommendations."""
    
    def __init__(self, guidelines_path):
        self.guidelines_path = guidelines_path
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.chunks = []
        self.index = None
        
    def build_index(self):
        """Build FAISS index from guidelines."""
        
        guidelines_text = load_guidelines_pdf(self.guidelines_path)
        
        if not guidelines_text:
            guidelines_text = self._get_fallback_guidelines()
        
        self.chunks = chunk_text(guidelines_text, chunk_size=400, overlap=50)
        
        # Create embeddings
        embeddings = self.embedder.encode(self.chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        return len(self.chunks)
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve most relevant guideline chunks.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant text chunks
        """
        if not self.index:
            self.build_index()
        
        # Encode query
        query_embedding = self.embedder.encode([query])[0]
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        return [self.chunks[idx] for idx in indices[0]]
    
    def generate_recommendation(self, risk_assessment: Dict, 
                               conversation_text: str) -> Dict[str, str]:
        """
        Generate recommendations using RAG.
        
        Args:
            risk_assessment: Risk scores and categories
            conversation_text: Original conversation
            
        Returns:
            Dictionary with recommendations and treatment plans
        """
        if not self.client:
            return {
                'hiv_recommendation': 'API not configured',
                'mh_recommendation': 'API not configured',
                'integrated_plan': 'API not configured'
            }
        
        # Retrieve relevant guidelines
        hiv_query = f"HIV risk {risk_assessment['hiv_risk']['final_category']} testing PrEP treatment"
        mh_query = f"mental health {risk_assessment['mental_health_risk']['final_category']} counseling treatment"
        
        hiv_context = self.retrieve_relevant_chunks(hiv_query, top_k=3)
        mh_context = self.retrieve_relevant_chunks(mh_query, top_k=3)
        
        # Generate recommendations
        prompt = f"""You are a South African healthcare professional providing evidence-based recommendations according to NDOH guidelines.

RISK ASSESSMENT SUMMARY:
- HIV Risk: {risk_assessment['hiv_risk']['final_category']} (Score: {risk_assessment['hiv_risk']['final_score']})
- Mental Health Risk: {risk_assessment['mental_health_risk']['final_category']} (Score: {risk_assessment['mental_health_risk']['final_score']})

HIV RISK FACTORS:
{', '.join(risk_assessment['hiv_risk']['llm_based'].get('risk_factors', ['None identified']))}

MENTAL HEALTH CONCERNS:
{', '.join(risk_assessment['mental_health_risk']['llm_based'].get('risk_factors', ['None identified']))}

RELEVANT NDOH GUIDELINES - HIV:
{chr(10).join(hiv_context)}

RELEVANT NDOH GUIDELINES - MENTAL HEALTH:
{chr(10).join(mh_context)}

CONVERSATION CONTEXT:
{conversation_text[:800]}...

Provide:
1. HIV RECOMMENDATION: Specific evidence-based actions (testing, PrEP, counseling) per SA guidelines
2. MENTAL HEALTH RECOMMENDATION: Appropriate interventions and referrals per SA guidelines  
3. INTEGRATED TREATMENT PLAN: Holistic 3-step action plan addressing both concerns

Format as JSON:
{{
  "hiv_recommendation": "...",
  "mh_recommendation": "...",
  "integrated_plan": "..."
}}"""

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=1500,
                temperature=0.5,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            result_text = response.choices[0].message.content.strip()

            import json 
            import re
            result_text = re.sub(r'^```json\s*|\s*```$', '', result_text, flags=re.MULTILINE)
            recommendations = json.loads(result_text)
            
            return recommendations
            
        except Exception as e:
            return {
                'hiv_recommendation': f'Error generating recommendation: {str(e)}',
                'mh_recommendation': f'Error generating recommendation: {str(e)}',
                'integrated_plan': f'Error generating plan: {str(e)}'
            }
    
    def _get_fallback_guidelines(self) -> str:
        """Fallback guidelines when PDF unavailable."""
        return """
        SOUTH AFRICAN NATIONAL DEPARTMENT OF HEALTH HIV GUIDELINES:
        
        HIV Testing: Universal testing recommended for all sexually active individuals. 
        Routine testing every 3-6 months for high-risk populations including sex workers, 
        MSM, people who inject drugs, and those with multiple partners.
        
        PrEP (Pre-Exposure Prophylaxis): Recommended for HIV-negative individuals at 
        substantial risk. Daily oral tenofovir-based PrEP reduces infection risk by >90%.
        Eligible groups include serodiscordant couples, sex workers, MSM, and those 
        with recent STI diagnosis.
        
        PEP (Post-Exposure Prophylaxis): Must be initiated within 72 hours of potential
        exposure. 28-day course of antiretroviral therapy. Available at all public facilities.
        
        STI Management: Syndromic management approach. Partner notification and treatment 
        essential. Regular screening for high-risk groups.
        
        MENTAL HEALTH GUIDELINES:
        
        Primary Mental Healthcare: Integration of mental health services into primary care.
        PHC nurses trained to identify and manage common mental disorders.
        
        Depression/Anxiety: Counseling, psychosocial support, and medication where appropriate.
        Referral to mental health specialists for moderate-severe cases.
        
        Crisis Intervention: Immediate assessment for suicidal ideation or psychosis.
        24/7 crisis helplines available. Psychiatric emergency services at district hospitals.
        
        Community-Based Care: Home-based care teams, support groups, and peer counseling 
        programs. Family involvement encouraged.
        
        Integrated Care: Screening for mental health in HIV clinics and vice versa. 
        Holistic patient-centered approach addressing comorbidities.
        """