"""
Text preprocessing module for Arabic and French text normalization.

This module provides functions to clean and normalize text before
generating embeddings for semantic similarity search.
"""

import re
import unicodedata
from typing import Optional


# Arabic diacritics (Harakat) - to be removed
ARABIC_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670]')

# Arabic Tatweel (Kashida) - elongation character
ARABIC_TATWEEL = '\u0640'

# Arabic Alif forms mapping (normalize to bare Alif)
ARABIC_ALIF_FORMS = {
    '\u0623': '\u0627',  # أ (Alif with Hamza above) -> ا
    '\u0625': '\u0627',  # إ (Alif with Hamza below) -> ا  
    '\u0622': '\u0627',  # آ (Alif with Madda) -> ا
    '\u0671': '\u0627',  # ٱ (Alif Wasla) -> ا
}

# Arabic Ya forms mapping (normalize to dotless Ya)
ARABIC_YA_FORMS = {
    '\u064A': '\u0649',  # ي (Ya with dots) -> ى (Alif Maksura)
}

# HTML tag pattern
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')

# Multiple whitespace pattern
MULTIPLE_WHITESPACE = re.compile(r'\s+')

# URL pattern
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')

# Email pattern  
EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text.
    
    Args:
        text: Input text potentially containing HTML tags.
        
    Returns:
        Text with HTML tags removed.
    """
    return HTML_TAG_PATTERN.sub(' ', text)


def remove_urls(text: str) -> str:
    """Remove URLs from text.
    
    Args:
        text: Input text potentially containing URLs.
        
    Returns:
        Text with URLs removed.
    """
    return URL_PATTERN.sub(' ', text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text.
    
    Args:
        text: Input text potentially containing email addresses.
        
    Returns:
        Text with email addresses removed.
    """
    return EMAIL_PATTERN.sub(' ', text)


def normalize_whitespace(text: str) -> str:
    """Normalize multiple whitespaces to single space and strip.
    
    Args:
        text: Input text with potential excess whitespace.
        
    Returns:
        Text with normalized whitespace.
    """
    return MULTIPLE_WHITESPACE.sub(' ', text).strip()


def remove_arabic_diacritics(text: str) -> str:
    """Remove Arabic diacritical marks (Harakat).
    
    Removes vowel marks like Fatha, Damma, Kasra, Sukun, Shadda, etc.
    
    Args:
        text: Arabic text with potential diacritics.
        
    Returns:
        Text with diacritics removed.
    """
    return ARABIC_DIACRITICS.sub('', text)


def remove_arabic_tatweel(text: str) -> str:
    """Remove Arabic Tatweel (Kashida) elongation character.
    
    Args:
        text: Arabic text with potential Tatweel characters.
        
    Returns:
        Text with Tatweel removed.
    """
    return text.replace(ARABIC_TATWEEL, '')


def normalize_arabic_alif(text: str) -> str:
    """Normalize different Alif forms to bare Alif.
    
    Converts أ, إ, آ, ٱ -> ا
    
    Args:
        text: Arabic text with various Alif forms.
        
    Returns:
        Text with normalized Alif characters.
    """
    for old, new in ARABIC_ALIF_FORMS.items():
        text = text.replace(old, new)
    return text


def normalize_arabic_ya(text: str) -> str:
    """Normalize Ya with dots to Alif Maksura (dotless Ya).
    
    Converts ي -> ى
    
    Args:
        text: Arabic text with Ya characters.
        
    Returns:
        Text with normalized Ya characters.
    """
    for old, new in ARABIC_YA_FORMS.items():
        text = text.replace(old, new)
    return text


def normalize_arabic(text: str) -> str:
    """Apply all Arabic text normalizations.
    
    Applies the following normalizations in order:
    1. Remove diacritics (Harakat)
    2. Remove Tatweel (Kashida)
    3. Normalize Alif forms
    4. Normalize Ya forms
    
    Args:
        text: Arabic text to normalize.
        
    Returns:
        Normalized Arabic text.
    """
    text = remove_arabic_diacritics(text)
    text = remove_arabic_tatweel(text)
    text = normalize_arabic_alif(text)
    text = normalize_arabic_ya(text)
    return text


def normalize_french(text: str) -> str:
    """Apply French text normalizations.
    
    Currently applies:
    1. Lowercase conversion
    
    Note: LaBSE is robust to French accents, so aggressive
    accent stripping is not strictly necessary.
    
    Args:
        text: French text to normalize.
        
    Returns:
        Normalized French text.
    """
    return text.lower()


def detect_primary_script(text: str) -> str:
    """Detect the primary script used in text.
    
    Simple heuristic: counts Arabic vs Latin characters.
    
    Args:
        text: Input text.
        
    Returns:
        'arabic' if primarily Arabic, 'latin' if primarily Latin,
        'mixed' if roughly equal, 'unknown' if neither.
    """
    arabic_count = 0
    latin_count = 0
    
    for char in text:
        if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F':
            arabic_count += 1
        elif 'a' <= char.lower() <= 'z':
            latin_count += 1
    
    total = arabic_count + latin_count
    if total == 0:
        return 'unknown'
    
    arabic_ratio = arabic_count / total
    
    if arabic_ratio > 0.6:
        return 'arabic'
    elif arabic_ratio < 0.4:
        return 'latin'
    else:
        return 'mixed'


def clean_text(text: str) -> str:
    """Apply general text cleaning.
    
    Applies the following cleanings:
    1. Remove HTML tags
    2. Remove URLs
    3. Remove email addresses
    4. Normalize whitespace
    
    Args:
        text: Raw input text.
        
    Returns:
        Cleaned text.
    """
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = normalize_whitespace(text)
    return text


def normalize_text(text: str, force_lowercase: bool = False) -> str:
    """Full text normalization pipeline.
    
    Applies all cleaning and normalization steps:
    1. General cleaning (HTML, URLs, emails, whitespace)
    2. Arabic-specific normalization (if Arabic text detected)
    3. French/Latin-specific normalization (if Latin text detected)
    
    Args:
        text: Raw input text (French, Arabic, or mixed).
        force_lowercase: If True, always lowercase (default: False,
            only lowercase if primarily Latin text).
        
    Returns:
        Fully normalized text ready for embedding.
    """
    if not text:
        return ""
    
    # Apply general cleaning
    text = clean_text(text)
    
    if not text:
        return ""
    
    # Detect primary script
    script = detect_primary_script(text)
    
    # Apply script-specific normalization
    if script == 'arabic':
        text = normalize_arabic(text)
    elif script == 'latin':
        text = normalize_french(text)
    elif script == 'mixed':
        # Apply both normalizations for mixed text
        text = normalize_arabic(text)
        text = normalize_french(text)
    
    # Apply forced lowercase if requested
    if force_lowercase:
        text = text.lower()
    
    # Final whitespace normalization
    text = normalize_whitespace(text)
    
    return text


def preprocess_batch(texts: list[str], force_lowercase: bool = False) -> list[str]:
    """Preprocess a batch of texts.
    
    Args:
        texts: List of raw input texts.
        force_lowercase: If True, always lowercase.
        
    Returns:
        List of normalized texts.
    """
    return [normalize_text(text, force_lowercase) for text in texts]


