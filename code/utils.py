import os
from typing import Optional
from dotenv import load_dotenv

def load_hf_token() -> Optional[str]:
    """
    Load HuggingFace token from environment variables or .env file.
    Returns None if token is not found.
    """
    load_dotenv()  # Load environment variables from .env file
    return os.getenv("HUGGINGFACE_TOKEN")

def load_openai_token() -> Optional[str]:
    """
    Load OpenAI API token from environment variables or .env file.
    Returns None if token is not found.
    """
    load_dotenv()  # Load environment variables from .env file
    return os.getenv("OPENAI_API_KEY")

def validate_dataset_type(dataset_type: str) -> bool:
    """Validate if the dataset type is supported."""
    return dataset_type in ["oncqa", "askadocs", "usmle"]

def validate_perturbation_type(perturbation_type: str) -> bool:
    """Validate if the perturbation type is supported."""
    return perturbation_type in ["gender", "stylistic", "viewpoint"]

def validate_gender_variant(variant: str) -> bool:
    """Validate if the gender variant is supported."""
    return variant in ["swap", "remove"]

def validate_stylistic_variant(variant: str) -> bool:
    """Validate if the stylistic variant is supported."""
    return variant in ["uncertain", "colorful"]

def validate_viewpoint_variant(variant: str) -> bool:
    """Validate if the viewpoint variant is supported."""
    return variant in ["multiturn", "summarized"]

def setup_logging():
    """Setup basic logging configuration."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__) 