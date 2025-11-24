import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONVERSATIONS_FILE = DATA_DIR / "health_ai_whatsapp_100_conversations_long.txt"
GUIDELINES_PDF = DATA_DIR / "NDoH-guidelines.pdf" 

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4.1"

# Risk thresholds for rule-based scoring
HIV_RISK_KEYWORDS = {
    "high": ["unprotected", "multiple partners", "sti", "sexually transmitted", 
             "injection drug", "needle sharing", "sex work", "recent exposure"],
    "medium": ["partner", "condom", "test", "worried", "concerned", "symptoms"],
    "low": ["healthy", "monogamous", "protected", "negative test"]
}

MENTAL_HEALTH_KEYWORDS = {
    "high": ["suicide", "self-harm", "hopeless", "worthless", "can't go on",
             "death wish", "ending it", "severe depression", "psychosis"],
    "medium": ["depressed", "anxious", "stressed", "worried", "can't sleep",
               "panic", "trauma", "abuse", "isolated", "crying"],
    "low": ["concerned", "nervous", "tired", "overwhelmed"]
}

# Symptom patterns for hybrid scoring
HIV_SYMPTOM_PATTERNS = [
    "fever", "night sweats", "weight loss", "fatigue", "swollen glands",
    "rash", "sore throat", "muscle aches", "genital sores", "discharge"
]

MH_SYMPTOM_PATTERNS = [
    "sad", "hopeless", "anxious", "panic", "can't focus", "irritable",
    "mood swings", "hearing voices", "paranoid", "flashbacks"
]