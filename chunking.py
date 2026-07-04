from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_community.docstore.document import Document


def process_text_chunks(
    text: str, chunk_size: int = 2048, chunk_overlap: int = 0
) -> List[str]:
    """
    Split the input text into chunks using the RecursiveCharacterTextSplitter.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The target size of each chunk (default: 512).
        chunk_overlap (int): The overlap between chunks (default: 0).

    Returns:
        List[str]: A list of text chunks.
    """
    # Model name
    model_name = "gpt-4o"
    try:
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Split the text
        chunks = text_splitter.split_text(text)

        return chunks

    except Exception as e:
        print(f"Error splitting text: {e}")
        return []


def process_text_chunks_with_pages(
    documents: List[Document] , chunk_size: int = 2048, chunk_overlap: int = 0
) -> List[Document]:
    """
    Split the input text into chunks using the RecursiveCharacterTextSplitter.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The target size of each chunk (default: 512).
        chunk_overlap (int): The overlap between chunks (default: 0).

    Returns:
        List[str]: A list of text chunks.
    """
    # Model name
    model_name = "gpt-4o"
    try:
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Split the text
        chunks = text_splitter.split_documents(documents)

        return chunks

    except Exception as e:
        print(f"Error splitting text: {e}")
        return []


# Example usage
if __name__ == "__main__":
    # Example text
    sample_text = "This is a sample text that will be split into chunks. " * 50

    # Chunk the text
    chunks = process_text_chunks(text=sample_text, chunk_size=512, chunk_overlap=0)

    # Print the chunks
    for idx, chunk in enumerate(chunks, start=1):
        print(f"Chunk {idx}:\n{chunk}\n{'-'*40}")
