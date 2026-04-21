"""Context / adaptation layer — RAG, glossaries, TM, prompt building."""
from .glossary import Glossary
from .prompt_builder import PromptBuilder
from .rag import HybridRetriever, RetrievalResult
from .translation_memory import TranslationMemory

__all__ = [
    "Glossary",
    "HybridRetriever",
    "PromptBuilder",
    "RetrievalResult",
    "TranslationMemory",
]
