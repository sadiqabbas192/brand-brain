from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .gemini import get_gemini_client

# Initialize text splitter with v1 settings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def chunk_text(text: str) -> List[str]:
    """Splits text into chunks using RecursiveCharacterTextSplitter."""
    return text_splitter.split_text(text)

def generate_embedding(text: str, title: str = "Brand Asset", task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """Generates 768-dim embeddings using gemini-embedding-001."""
    client = get_gemini_client()
    try:
        # Note: 'title' is only valid for RETRIEVAL_DOCUMENT task type
        config = {
            'output_dimensionality': 768,
            'task_type': task_type
        }
        if task_type == 'RETRIEVAL_DOCUMENT':
            config['title'] = title

        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=config
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"Embedding Error: {e}")
        return []

def generate_query_embedding(text: str) -> List[float]:
    """Helper for query embeddings (RETRIEVAL_QUERY)."""
    return generate_embedding(text, task_type="RETRIEVAL_QUERY")
