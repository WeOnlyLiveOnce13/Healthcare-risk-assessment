import re
from typing import List, Dict
from pathlib import Path
from pypdf import PdfReader


def load_conversations(file_path: Path) -> List[Dict[str, str]]:
    """
    Load and parse conversations from text file.
    
    Args:
        file_path: Path to conversations file
        
    Returns:
        List of conversation dictionaries with metadata
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by conversation separator
    conv_blocks = content.split("========== Conversation ==========")
    conversations = []
    
    for block in conv_blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        messages = []
        
        for line in lines:
            # Parse timestamp and message
            match = re.match(r'\[([^\]]+)\] (User|AI): (.+)', line)
            if match:
                timestamp, role, text = match.groups()
                messages.append({
                    'timestamp': timestamp,
                    'role': role,
                    'text': text.strip()
                })
        
        if messages:
            # Combine all text for analysis
            full_text = ' '.join([m['text'] for m in messages])
            user_text = ' '.join([m['text'] for m in messages if m['role'] == 'User'])
            
            conversations.append({
                'messages': messages,
                'full_text': full_text,
                'user_text': user_text,
                'message_count': len(messages)
            })
    
    return conversations


def load_guidelines_pdf(pdf_path: Path) -> str:
    """
    Extract text from SA NDOH guidelines PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text content
    """
    if not pdf_path.exists():
        return ""
    
    reader = PdfReader(pdf_path)
    text_content = []
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_content.append(text)
    
    return "\n\n".join(text_content)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for RAG.
    
    Args:
        text: Input text
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks