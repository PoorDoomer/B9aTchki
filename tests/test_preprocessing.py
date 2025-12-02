"""
Unit tests for the text preprocessing module.
Tests Arabic and French text normalization functions.
"""

import pytest
from src.preprocessing import (
    remove_html_tags,
    remove_urls,
    remove_emails,
    normalize_whitespace,
    remove_arabic_diacritics,
    remove_arabic_tatweel,
    normalize_arabic_alif,
    normalize_arabic_ya,
    normalize_arabic,
    normalize_french,
    detect_primary_script,
    clean_text,
    normalize_text,
    preprocess_batch,
)


class TestGeneralCleaning:
    """Tests for general text cleaning functions."""
    
    def test_remove_html_tags(self):
        """Test HTML tag removal."""
        text = "<p>Hello <b>World</b></p>"
        result = remove_html_tags(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Hello" in result
        assert "World" in result
    
    def test_remove_html_tags_empty(self):
        """Test HTML removal on empty string."""
        assert remove_html_tags("") == ""
    
    def test_remove_html_tags_no_tags(self):
        """Test HTML removal when no tags present."""
        text = "Plain text without tags"
        assert remove_html_tags(text) == text
    
    def test_remove_urls_http(self):
        """Test HTTP URL removal."""
        text = "Visit http://example.com for more"
        result = remove_urls(text)
        assert "http://example.com" not in result
        assert "Visit" in result
        assert "for more" in result
    
    def test_remove_urls_https(self):
        """Test HTTPS URL removal."""
        text = "Visit https://example.com/path for more"
        result = remove_urls(text)
        assert "https://" not in result
    
    def test_remove_urls_www(self):
        """Test www URL removal."""
        text = "Visit www.example.com for more"
        result = remove_urls(text)
        assert "www.example.com" not in result
    
    def test_remove_emails(self):
        """Test email address removal."""
        text = "Contact support@example.com for help"
        result = remove_emails(text)
        assert "support@example.com" not in result
        assert "Contact" in result
        assert "for help" in result
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "  Multiple   spaces   here  "
        result = normalize_whitespace(text)
        assert result == "Multiple spaces here"
    
    def test_normalize_whitespace_newlines(self):
        """Test newline to space conversion."""
        text = "Line one\nLine two\tLine three"
        result = normalize_whitespace(text)
        assert "\n" not in result
        assert "\t" not in result
        assert "Line one Line two Line three" == result


class TestArabicNormalization:
    """Tests for Arabic-specific text normalization."""
    
    def test_remove_diacritics_fatha(self):
        """Test removal of Fatha diacritic."""
        # Arabic text with Fatha (short 'a' vowel)
        text = "كَتَبَ"  # kataba
        result = remove_arabic_diacritics(text)
        assert "كتب" == result
    
    def test_remove_diacritics_mixed(self):
        """Test removal of multiple diacritics."""
        # Text with various diacritics
        text = "الْعَرَبِيَّةُ"  # al-arabiyyatu
        result = remove_arabic_diacritics(text)
        # Should only have consonants left
        assert "ْ" not in result  # Sukun
        assert "َ" not in result  # Fatha
        assert "ِ" not in result  # Kasra
        assert "ُ" not in result  # Damma
        assert "ّ" not in result  # Shadda
    
    def test_remove_tatweel(self):
        """Test Tatweel (Kashida) removal."""
        # Tatweel is ـ (U+0640)
        text = "مـــرحـــبا"  # marhabaa with tatweel
        result = remove_arabic_tatweel(text)
        assert "ـ" not in result
        assert "مرحبا" == result
    
    def test_normalize_alif_hamza_above(self):
        """Test Alif with Hamza above normalization."""
        text = "أحمد"  # Ahmad
        result = normalize_arabic_alif(text)
        assert result[0] == "ا"  # Bare Alif
    
    def test_normalize_alif_hamza_below(self):
        """Test Alif with Hamza below normalization."""
        text = "إسلام"  # Islam
        result = normalize_arabic_alif(text)
        assert result[0] == "ا"  # Bare Alif
    
    def test_normalize_alif_madda(self):
        """Test Alif with Madda normalization."""
        text = "آمين"  # Amin
        result = normalize_arabic_alif(text)
        assert result[0] == "ا"  # Bare Alif
    
    def test_normalize_ya(self):
        """Test Ya normalization to Alif Maksura."""
        text = "في"  # fi (in)
        result = normalize_arabic_ya(text)
        assert "ى" in result  # Alif Maksura
    
    def test_normalize_arabic_complete(self):
        """Test complete Arabic normalization pipeline."""
        # Text with diacritics, tatweel, various alif forms
        text = "أَنَا أُحِبُّ الْعَرَبِيَّةَ"
        result = normalize_arabic(text)
        # Should be simplified
        assert "ا" in result
        assert "ْ" not in result
        assert "َ" not in result


class TestFrenchNormalization:
    """Tests for French-specific text normalization."""
    
    def test_normalize_french_lowercase(self):
        """Test French text is lowercased."""
        text = "Bonjour Le Monde"
        result = normalize_french(text)
        assert result == "bonjour le monde"
    
    def test_normalize_french_accents_preserved(self):
        """Test that French accents are preserved."""
        text = "Café Résumé Naïve"
        result = normalize_french(text)
        assert "é" in result
        assert "ï" in result


class TestScriptDetection:
    """Tests for script detection function."""
    
    def test_detect_arabic_script(self):
        """Test detection of Arabic text."""
        text = "مرحبا بالعالم"
        result = detect_primary_script(text)
        assert result == "arabic"
    
    def test_detect_latin_script(self):
        """Test detection of Latin/French text."""
        text = "Bonjour le monde"
        result = detect_primary_script(text)
        assert result == "latin"
    
    def test_detect_mixed_script(self):
        """Test detection of mixed Arabic/Latin text."""
        # Roughly equal amounts
        text = "مرحبا Hello World"
        result = detect_primary_script(text)
        # Could be mixed or one of the scripts
        assert result in ["arabic", "latin", "mixed"]
    
    def test_detect_empty_text(self):
        """Test detection of empty text."""
        result = detect_primary_script("")
        assert result == "unknown"
    
    def test_detect_numbers_only(self):
        """Test detection of numbers only."""
        result = detect_primary_script("12345")
        assert result == "unknown"


class TestCleanText:
    """Tests for the clean_text function."""
    
    def test_clean_text_combined(self):
        """Test cleaning with multiple elements to remove."""
        text = "<p>Visit http://test.com or email test@test.com</p>"
        result = clean_text(text)
        assert "<p>" not in result
        assert "http://test.com" not in result
        assert "test@test.com" not in result
        assert "Visit" in result


class TestNormalizeText:
    """Tests for the main normalize_text function."""
    
    def test_normalize_french_text(self):
        """Test full normalization of French text."""
        text = "  <p>Ma connexion Internet ne fonctionne pas!</p>  "
        result = normalize_text(text)
        assert result == "ma connexion internet ne fonctionne pas!"
    
    def test_normalize_arabic_text(self):
        """Test full normalization of Arabic text."""
        text = "الإنترنت مقطوع"
        result = normalize_text(text)
        assert len(result) > 0
        # Should have normalized Alif
        assert "ا" in result
    
    def test_normalize_empty_text(self):
        """Test normalization of empty text."""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""
    
    def test_normalize_force_lowercase(self):
        """Test force_lowercase parameter."""
        text = "HELLO WORLD"
        result = normalize_text(text, force_lowercase=True)
        assert result == "hello world"


class TestPreprocessBatch:
    """Tests for batch preprocessing."""
    
    def test_preprocess_batch_french(self, sample_french_texts):
        """Test batch preprocessing of French texts."""
        results = preprocess_batch(sample_french_texts)
        assert len(results) == len(sample_french_texts)
        # All should be lowercased
        for result in results:
            assert result == result.lower()
    
    def test_preprocess_batch_arabic(self, sample_arabic_texts):
        """Test batch preprocessing of Arabic texts."""
        results = preprocess_batch(sample_arabic_texts)
        assert len(results) == len(sample_arabic_texts)
        # All should be non-empty
        for result in results:
            assert len(result) > 0
    
    def test_preprocess_batch_empty(self):
        """Test batch preprocessing with empty list."""
        results = preprocess_batch([])
        assert results == []


