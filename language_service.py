import logging
import os
import unicodedata
from typing import Optional

import fasttext

logger = logging.getLogger(__name__)

# Language map (extend as needed)
LANGUAGE_CODES = {
    "english": "en",
    "arabic": "ar",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "polish": "pl",
    "portuguese": "pt",
    "russian": "ru",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "hindi": "hi",
}

# Reverse map: ISO 639-1 code -> Language name (title case)
CODE_TO_LANGUAGE = {v: k.title() for k, v in LANGUAGE_CODES.items()}

# Right-to-left languages we explicitly support for directional isolation.
RTL_LANGUAGE_CODES = {"ar", "he", "fa", "ur"}


def _load_fasttext_model():
    model_path = os.getenv("FASTTEXT_MODEL_PATH", "/app/models/lid.176.bin")
    legacy_model_path = os.path.join(os.getenv("DATA_DIR", "/data"), "lid.176.bin")

    # One-time compatibility migration: move old /data model into the new model path.
    if (
        model_path != legacy_model_path
        and not os.path.exists(model_path)
        and os.path.exists(legacy_model_path)
    ):
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.replace(legacy_model_path, model_path)
            logger.info(
                f"Moved fastText model from {legacy_model_path} to {model_path}"
            )
        except Exception as e:
            logger.warning(
                f"Could not move fastText model to {model_path}: {e}. "
                f"Using legacy path {legacy_model_path}."
            )
            model_path = legacy_model_path

    if not os.path.exists(model_path):
        logger.error(f"fastText model not found at {model_path}. Please download it.")
        return None

    try:
        model = fasttext.load_model(model_path)
        logger.info("fastText language model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load fastText model: {e}")
        return None


MODEL = _load_fasttext_model()


def detect_language(text: str, expected_language: str = None) -> Optional[str]:
    """
    Detect language using fastText. Returns ISO 639-1 code (e.g., 'en', 'es').

    If expected_language is provided (e.g., "English", "Arabic"), and that language
    appears in the top predictions with probability reasonably close to the top result,
    prefer the expected language. This reduces false positives on short/ambiguous text.
    """
    if not MODEL:
        return None

    try:
        # fastText expects one line at a time - replace newlines with spaces
        text_single_line = text.replace("\n", " ").replace("\r", " ").strip()

        # Get predictions (top 5 for better accuracy)
        predictions = MODEL.predict(text_single_line, k=5)
        labels = predictions[0]  # list of '__label__xx'
        probs = predictions[1]  # corresponding probabilities

        # Strip '__label__' prefix
        codes = [label.replace("__label__", "") for label in labels]

        # Prefer expected language when it is reasonably close to the top prediction.
        if expected_language:
            expected_code = LANGUAGE_CODES.get(expected_language.lower())
            if expected_code:
                top_code = codes[0] if len(codes) > 0 else None
                top_prob = float(probs[0]) if len(probs) > 0 else 0.0

                if top_code == expected_code:
                    logger.debug(
                        f"Language match: expected {expected_language} is already top prediction"
                    )
                    return expected_code

                for code, prob in zip(codes, probs):
                    prob = float(prob)
                    # Use expected language if it is within a reasonable margin.
                    if code == expected_code and top_prob > 0 and (prob / top_prob) >= 0.60:
                        logger.debug(
                            f"Language override by channel context: expected {expected_language} "
                            f"({prob:.1%}) vs top {top_code} ({top_prob:.1%})"
                        )
                        return expected_code

        # Standard detection: take the top prediction
        detected_code = codes[0]
        detected_lang = CODE_TO_LANGUAGE.get(detected_code, "Unknown")
        if expected_language:
            logger.debug(
                f"Language mismatch from expected {expected_language}: detected {detected_lang}"
            )
        else:
            logger.debug(f"Language detected: {detected_lang}")
        return detected_code
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return None


def _language_to_code(language: Optional[str]) -> Optional[str]:
    if not language:
        return None

    normalized = language.strip().lower()
    if not normalized:
        return None

    if normalized in LANGUAGE_CODES:
        return LANGUAGE_CODES[normalized]

    # Allow callers to pass ISO 639-1 code directly.
    if len(normalized) == 2:
        return normalized

    return None


def _is_rtl_language(language: Optional[str]) -> bool:
    code = _language_to_code(language)
    return bool(code and code in RTL_LANGUAGE_CODES)


def _first_strong_direction(text: str) -> Optional[str]:
    """Return 'rtl' or 'ltr' using Unicode bidi classes of the first strong character."""
    for ch in text:
        bidi = unicodedata.bidirectional(ch)
        if bidi in {"R", "AL"}:
            return "rtl"
        if bidi == "L":
            return "ltr"
    return None


def apply_directional_isolation(
    text: str,
    target_language: Optional[str] = None,
    detected_language: Optional[str] = None,
) -> str:
    """
    Wrap text in Unicode RTL isolate markers when needed.

    Discord can visually reorder mixed Arabic + Latin/emoji text when no explicit
    directional context is provided. RLI/PDI keeps ordering stable.
    """
    if not text:
        return text

    # Avoid double-wrapping already isolated content.
    if text.startswith("\u2067") and text.endswith("\u2069"):
        return text

    prefer_rtl = _is_rtl_language(target_language) or (
        not target_language and _is_rtl_language(detected_language)
    )

    # Fallback: if metadata is missing, infer from first strong directional char.
    if not prefer_rtl and _first_strong_direction(text) == "rtl":
        prefer_rtl = True

    return f"\u2067{text}\u2069" if prefer_rtl else text
