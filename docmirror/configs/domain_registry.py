"""
Domain Registry — Document-type-specific identity field definitions.
=====================================================================

Maps each document type (e.g., bank_statement, invoice, contract) to a list
of identity field definitions. Each definition specifies:

    (display_name, candidate_key_1, candidate_key_2, ...)

The ``resolve_identity()`` function looks up fields for a given domain,
then searches the provided entities dict for the first matching candidate
key that has a non-empty value. This allows flexible extraction from
documents where the same concept may appear under different key names
(e.g., "Account holder", "Card holder", "Customer name" all map to
the "account_holder" identity field).

The wildcard domain ``"*"`` serves as a fallback for unrecognized
document types, providing minimal identity extraction (title, date, author).

Usage::

    from docmirror.configs.domain_registry import resolve_identity

    identity = resolve_identity("bank_statement", extracted_entities)
    # {'document_type': 'bank_statement', 'institution': 'HSBC', ...}
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

logger = logging.getLogger(__name__)

_CONFIGS_DIR = Path(__file__).parent


# ══════════════════════════════════════════════════════════════════════════════
# Multilingual Key Synonyms — loaded from key_synonyms.yaml
# ══════════════════════════════════════════════════════════════════════════════

def _load_key_synonyms() -> Dict[str, str]:
    """
    Load and flatten the key_synonyms.yaml config into a single lookup dict.

    The YAML structure is ``domain → locale → {raw_key: canonical_key}``.
    This function flattens all levels into one dict for O(1) lookup at runtime.
    If the YAML file is missing or malformed, returns an empty dict and logs
    a warning (graceful degradation — English-key documents still work).
    """
    yaml_path = _CONFIGS_DIR / "key_synonyms.yaml"
    if not yaml_path.exists():
        logger.warning("key_synonyms.yaml not found at %s, key normalization disabled", yaml_path)
        return {}

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception as e:
        logger.warning("Failed to load key_synonyms.yaml: %s", e)
        return {}

    if not isinstance(raw, dict):
        return {}

    flat: Dict[str, str] = {}
    for domain, locales in raw.items():
        if not isinstance(locales, dict):
            continue
        for locale, mappings in locales.items():
            if not isinstance(mappings, dict):
                continue
            flat.update(mappings)

    logger.debug("Loaded %d key synonyms from key_synonyms.yaml", len(flat))
    return flat


# Module-level singleton — loaded once on first import
KEY_SYNONYMS: Dict[str, str] = _load_key_synonyms()


def normalize_entity_keys(entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize locale-specific entity keys to canonical English equivalents.

    Applies ``KEY_SYNONYMS`` (loaded from ``key_synonyms.yaml``) to translate
    raw extracted keys (e.g. Chinese ``"账号"``) into canonical English keys
    (e.g. ``"Account number"``).

    Rules:
        - If the canonical key already exists in the dict, the original
          value is preserved (extraction-level data takes priority).
        - Unknown keys pass through unchanged.
        - The original dict is not mutated; a new dict is returned.

    Args:
        entities: Raw entity dict from document extraction.

    Returns:
        New dict with normalized keys.
    """
    normalized: Dict[str, Any] = {}

    for key, value in entities.items():
        canonical = KEY_SYNONYMS.get(key, key)
        # Don't overwrite if the canonical key was already set
        if canonical not in normalized:
            normalized[canonical] = value
        elif key not in KEY_SYNONYMS:
            # Original (non-synonym) key takes priority over synonym-derived
            normalized[key] = value

    return normalized


# ══════════════════════════════════════════════════════════════════════════════
# Document type → Identity Field Definitions
# ══════════════════════════════════════════════════════════════════════════════

# Each tuple: (display_name, candidate_key_1, candidate_key_2, ...)
# The resolver tries candidate keys in order and uses the first non-empty match.
DOMAIN_IDENTITY: Dict[str, List[Tuple[str, ...]]] = {
    "bank_statement": [
        ("institution", "bank_name", "Bank name", "Bank branch"),
        ("account_holder", "Account name", "Account holder", "Card holder", "Customer name"),
        ("account_number", "Account number", "Card number", "Account", "Customer account number"),
        ("query_period", "Query period", "Period", "From/to date"),
        ("currency", "Currency"),
        ("print_date", "Print date"),
    ],
    "invoice": [
        ("supplier", "Supplier", "Seller", "Invoice issuer"),
        ("buyer", "Buyer", "Buyer", "Invoice receiver"),
        ("invoice_no", "Invoice number", "Invoice number", "Invoice No"),
        ("amount", "Amount", "Total amount", "Total with tax"),
        ("tax", "Tax amount", "Tax rate"),
        ("date", "Invoice date", "Date"),
    ],
    "contract": [
        ("party_a", "Party A", "Party A"),
        ("party_b", "Party B", "Party B"),
        ("contract_no", "Contract number", "Contract No"),
        ("sign_date", "Signing date", "Signing date"),
        ("amount", "Contract amount", "Amount"),
    ],
    "receipt": [
        ("merchant", "Merchant name", "Merchant", "Merchant"),
        ("amount", "Transaction amount", "Amount", "Amount"),
        ("date", "Transaction date", "Date", "Date"),
    ],
    # Wildcard fallback — used for any unrecognized document type
    "*": [
        ("title", "Title", "Title"),
        ("date", "Date", "Date"),
        ("author", "Author", "Author"),
    ],
}


def resolve_identity(domain: str, entities: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract standardized identity fields from raw entities by document type.

    Internally normalizes entity keys via ``normalize_entity_keys()`` before
    matching, so locale-specific keys (e.g. Chinese) are resolved
    transparently.

    Args:
        domain:   Document type string (e.g., "bank_statement", "invoice").
        entities: Dict of extracted key-value entities from the document.

    Returns:
        Dict with standardized identity fields. Always includes
        ``"document_type"`` as the first key. Missing fields are set
        to empty strings.
    """
    # Normalize locale-specific keys to canonical English
    normalized = normalize_entity_keys(entities)

    fields = DOMAIN_IDENTITY.get(domain, DOMAIN_IDENTITY.get("*", []))
    identity: Dict[str, str] = {"document_type": domain}

    for field_def in fields:
        display_name = field_def[0]
        candidates = field_def[1:]
        # Try each candidate key in order; use the first non-empty value
        for key in candidates:
            val = normalized.get(key, "")
            if val:
                identity[display_name] = str(val)
                break
        else:
            # No candidate had a value — set to empty string
            identity[display_name] = ""

    return identity
