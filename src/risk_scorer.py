import re
import json
from typing import Dict, Tuple
from openai import OpenAI
from config import (
    OPENAI_API_KEY, LLM_MODEL,
    HIV_RISK_KEYWORDS, MENTAL_HEALTH_KEYWORDS,
    HIV_SYMPTOM_PATTERNS, MH_SYMPTOM_PATTERNS
)


class RiskScorer:
    """Hybrid risk scorer combining rule-based and LLM approaches."""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    
    def rule_based_score(self, text: str, keywords: Dict[str, list], 
                        symptoms: list) -> Tuple[float, str, Dict]:
        """
        Calculate risk score using keyword matching.
        
        Args:
            text: Conversation text
            keywords: Risk keywords by severity
            symptoms: List of symptom patterns
            
        Returns:
            Tuple of (score, category, details)
        """
        text_lower = text.lower()
        details = {
            'high_risk_matches': [],
            'medium_risk_matches': [],
            'low_risk_matches': [],
            'symptom_matches': []
        }
        
        # Count keyword matches
        high_count = sum(1 for kw in keywords['high'] if kw in text_lower)
        medium_count = sum(1 for kw in keywords['medium'] if kw in text_lower)
        low_count = sum(1 for kw in keywords['low'] if kw in text_lower)
        symptom_count = sum(1 for sym in symptoms if sym in text_lower)
        
        # Store matches
        details['high_risk_matches'] = [kw for kw in keywords['high'] if kw in text_lower]
        details['medium_risk_matches'] = [kw for kw in keywords['medium'] if kw in text_lower]
        details['low_risk_matches'] = [kw for kw in keywords['low'] if kw in text_lower]
        details['symptom_matches'] = [sym for sym in symptoms if sym in text_lower]
        
        # Calculate weighted score (0-1 scale)
        score = 0.0
        
        if high_count > 0:
            score += 0.7 + (high_count * 0.1)
        elif medium_count > 0:
            score += 0.4 + (medium_count * 0.05)
        elif low_count > 0:
            score += 0.1 + (low_count * 0.02)
        
        # Add symptom weight
        if symptom_count > 0:
            score += min(symptom_count * 0.1, 0.3)
        
        score = min(score, 1.0)
        
        # Categorize
        if score >= 0.7:
            category = "HIGH"
        elif score >= 0.4:
            category = "MEDIUM"
        else:
            category = "LOW"
        
        return score, category, details
    
    def llm_based_score(self, conversation_text: str, risk_type: str) -> Dict:
        """
        Use LLM to analyze conversation for risk assessment.
        
        Args:
            conversation_text: Full conversation text
            risk_type: Either 'HIV' or 'Mental Health'
            
        Returns:
            Dictionary with score, category, reasoning, and flags
        """
        if not self.client:
            return {
                'score': 0.0,
                'category': 'UNKNOWN',
                'reasoning': 'API key not configured',
                'risk_factors': [],
                'protective_factors': []
            }
        
        prompt = f"""You are a clinical risk assessment expert analyzing a conversation between an AI chatbot and a healthcare client.

Analyze the following conversation for {risk_type} RISK ASSESSMENT.

Conversation:
{conversation_text}

Provide a structured risk assessment in JSON format with:
1. "score": A numerical score from 0.0 (no risk) to 1.0 (extreme risk)
2. "category": Classification as "LOW", "MEDIUM", or "HIGH"
3. "reasoning": Brief explanation of the assessment (2-3 sentences)
4. "risk_factors": List of specific concerns identified
5. "protective_factors": List of positive/protective elements identified
6. "urgent_flags": Boolean indicating if immediate intervention needed

Consider:
- For HIV: Sexual behavior, exposure risks, symptoms, partner status, prevention practices
- For Mental Health: Mood indicators, self-harm mentions, functioning, support systems, trauma

Respond ONLY with valid JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.3,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            result_text = response.choices[0].message.content.strip()
            result_text = re.sub(r'^```json\s*|\s*```$', '', result_text, flags=re.MULTILINE)
            
            result = json.loads(result_text)
            return result
            
        except Exception as e:
            return {
                'score': 0.0,
                'category': 'ERROR',
                'reasoning': f'Analysis error: {str(e)}',
                'risk_factors': [],
                'protective_factors': [],
                'urgent_flags': False
            }
    
    def hybrid_score(self, conversation: Dict) -> Dict[str, Dict]:
        """
        Combine rule-based and LLM scoring for both risks.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            Dictionary with HIV and MH assessments
        """
        text = conversation['full_text']
        
        # Rule-based scores
        hiv_rule_score, hiv_rule_cat, hiv_details = self.rule_based_score(
            text, HIV_RISK_KEYWORDS, HIV_SYMPTOM_PATTERNS
        )
        mh_rule_score, mh_rule_cat, mh_details = self.rule_based_score(
            text, MENTAL_HEALTH_KEYWORDS, MH_SYMPTOM_PATTERNS
        )
        
        # LLM-based scores
        hiv_llm = self.llm_based_score(text, "HIV")
        mh_llm = self.llm_based_score(text, "Mental Health")
        
        # Combine scores (weighted average: 40% rule-based, 60% LLM)
        hiv_final_score = (hiv_rule_score * 0.4) + (hiv_llm.get('score', 0) * 0.6)
        mh_final_score = (mh_rule_score * 0.4) + (mh_llm.get('score', 0) * 0.6)
        
        # Final categorization
        def categorize(score):
            if score >= 0.7:
                return "HIGH"
            elif score >= 0.4:
                return "MEDIUM"
            return "LOW"
        
        return {
            'hiv_risk': {
                'final_score': round(hiv_final_score, 3),
                'final_category': categorize(hiv_final_score),
                'rule_based': {
                    'score': round(hiv_rule_score, 3),
                    'category': hiv_rule_cat,
                    'details': hiv_details
                },
                'llm_based': hiv_llm
            },
            'mental_health_risk': {
                'final_score': round(mh_final_score, 3),
                'final_category': categorize(mh_final_score),
                'rule_based': {
                    'score': round(mh_rule_score, 3),
                    'category': mh_rule_cat,
                    'details': mh_details
                },
                'llm_based': mh_llm
            }
        }