from typing import Dict, List
import pandas as pd
from risk_scorer import RiskScorer
from rag_recommender import RAGRecommender
from data_loader import load_conversations
from config import CONVERSATIONS_FILE, GUIDELINES_PDF


class ConversationAnalyzer:
    """Main analyzer for risk assessment and recommendations."""
    
    def __init__(self):
        self.risk_scorer = RiskScorer()
        self.recommender = RAGRecommender(GUIDELINES_PDF)
        self.recommender.build_index()
    
    def analyze_conversation(self, conversation: Dict) -> Dict:
        """
        Complete analysis of a single conversation.
        
        Args:
            conversation: Conversation dictionary
            
        Returns:
            Full analysis results
        """
        # Risk assessment
        risk_scores = self.risk_scorer.hybrid_score(conversation)
        
        # Generate recommendations
        recommendations = self.recommender.generate_recommendation(
            risk_scores, 
            conversation['full_text']
        )
        
        return {
            'conversation': {
                'message_count': conversation['message_count'],
                'text_preview': conversation['full_text'][:200] + '...'
            },
            'risk_assessment': risk_scores,
            'recommendations': recommendations
        }
    
    def analyze_dataset(self, limit: int = None) -> List[Dict]:
        """
        Analyze all conversations in dataset.
        
        Args:
            limit: Maximum number to analyze
            
        Returns:
            List of analysis results
        """
        conversations = load_conversations(CONVERSATIONS_FILE)
        
        if limit:
            conversations = conversations[:limit]
        
        results = []
        for i, conv in enumerate(conversations):
            print(f"Analyzing conversation {i+1}/{len(conversations)}...")
            result = self.analyze_conversation(conv)
            result['conversation_id'] = i + 1
            results.append(result)
        
        return results
    
    def create_summary_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """
        Create summary DataFrame from results.
        
        Args:
            results: List of analysis results
            
        Returns:
            Pandas DataFrame with key metrics
        """
        summary_data = []
        
        for r in results:
            summary_data.append({
                'conversation_id': r['conversation_id'],
                'message_count': r['conversation']['message_count'],
                'hiv_score': r['risk_assessment']['hiv_risk']['final_score'],
                'hiv_category': r['risk_assessment']['hiv_risk']['final_category'],
                'mh_score': r['risk_assessment']['mental_health_risk']['final_score'],
                'mh_category': r['risk_assessment']['mental_health_risk']['final_category'],
                'hiv_rule_score': r['risk_assessment']['hiv_risk']['rule_based']['score'],
                'hiv_llm_score': r['risk_assessment']['hiv_risk']['llm_based'].get('score', 0),
                'mh_rule_score': r['risk_assessment']['mental_health_risk']['rule_based']['score'],
                'mh_llm_score': r['risk_assessment']['mental_health_risk']['llm_based'].get('score', 0)
            })
        
        return pd.DataFrame(summary_data)