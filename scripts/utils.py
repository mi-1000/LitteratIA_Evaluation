from lingua import LanguageDetectorBuilder, Language

LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_all_languages().build()

def is_french(text: str, min_confidence = 0.0) -> bool:
    """
    Returns `True` if the text is in French, `False` otherwise.
    Optional parameter `min_confidence` can be set to a value between 0.0 and 1.0 to specify the minimum confidence level required for the text to be considered French.
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return False
    
    confidence_values = LANGUAGE_DETECTOR.compute_language_confidence_values(text)
    
    fr_confidence = next((cv.value for cv in confidence_values if cv.language == Language.FRENCH), 0.0)
    
    if not confidence_values:
        return False
        
    most_likely_language = confidence_values[0].language
    
    return most_likely_language == Language.FRENCH and fr_confidence >= min_confidence