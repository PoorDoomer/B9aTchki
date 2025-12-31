"""
Module focused on chunking functionality.

"""


import logging
import math


"""
Input = text string
Output = List of text chunks

"""
class Chunker:
    """
    Docstring for Chunker
    
    :var Args: Description
    :var text: Description
    :vartype text: Input
    :var Returns: Description
    """
    def __init__(self, chunk_size: int = 250, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = logging.getLogger(__name__)

    def chunk_text(self, text: str) -> list[str]:
        """
        Chunk the input text into smaller segments.

        Args:
            text: Input text string.

        Returns:
            List of text chunks.
        """
        if not text:
            return []

        words = text.split()
        total_words = len(words)
        chunks = []
        

        start = 0
        while start < total_words:
            end = min(start + self.chunk_size, total_words)
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            if end == total_words:
                break
            start += self.chunk_size - self.overlap

        self.logger.debug(f"Chunked text into {len(chunks)} chunks.")
        return chunks