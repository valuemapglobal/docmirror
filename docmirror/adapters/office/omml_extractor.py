"""
OMML Extractor: Office Math ML to LaTeX Converter
===================================================

Extracts and converts Office Math Format (OMML) directly from Word's
underlying XML structure into LaTeX strings.

This avoids the need for heavy visual math OCR models (like UniMERNet)
when the document is a pure electronic .docx file with embedded equations.
"""
from __future__ import annotations


import logging
from lxml import etree

logger = logging.getLogger(__name__)

# OMML Namespace
NS = {'m': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


class OMMLExtractor:
    """Converts OMML Elements (<m:oMath>) to LaTeX string representations."""

    @classmethod
    def convert_element(cls, omath_elem: etree._Element) -> str:
        """
        Convert a single <m:oMath> element to a LaTeX string.

        Args:
            omath_elem: The lxml Element representing <m:oMath>.

        Returns:
            A string containing the LaTeX representation.
        """
        try:
            return cls._parse_node(omath_elem)
        except Exception as e:
            logger.debug(f"[OMMLExtractor] Conversion error: {e}")
            return ""

    @classmethod
    def _parse_node(cls, node: etree._Element) -> str:
        """Recursively parsing OMML nodes into LaTeX."""
        if node is None:
            return ""

        tag = etree.QName(node.tag).localname
        
        # Base case: text nodes
        if tag == "t":
            return node.text or ""
            
        # Run node
        elif tag == "r":
            text = "".join(cls._parse_node(child) for child in node)
            return text
            
        # Fraction <m:f>
        elif tag == "f":
            num = cls._get_child(node, "fName") or cls._get_child(node, "num")
            den = cls._get_child(node, "fDir") or cls._get_child(node, "den")
            
            num_tex = cls._parse_node(num) if num is not None else ""
            den_tex = cls._parse_node(den) if den is not None else ""
            return f"\\frac{{{num_tex}}}{{{den_tex}}}"
            
        # Superscript <m:sSup>
        elif tag == "sSup":
            base = cls._parse_node(cls._get_child(node, "e"))
            sup = cls._parse_node(cls._get_child(node, "sup"))
            return f"{base}^{{{sup}}}"
            
        # Subscript <m:sSub>
        elif tag == "sSub":
            base = cls._parse_node(cls._get_child(node, "e"))
            sub = cls._parse_node(cls._get_child(node, "sub"))
            return f"{base}_{{{sub}}}"
            
        # Sub-Superscript <m:sSubSup>
        elif tag == "sSubSup":
            base = cls._parse_node(cls._get_child(node, "e"))
            sub = cls._parse_node(cls._get_child(node, "sub"))
            sup = cls._parse_node(cls._get_child(node, "sup"))
            return f"{base}_{{{sub}}}^{{{sup}}}"

        # Mathematical equation root
        elif tag == "oMath":
            return "".join(cls._parse_node(child) for child in node)
            
        # Generic fallback: just concatenate children
        else:
            return "".join(cls._parse_node(child) for child in node)

    @classmethod
    def _get_child(cls, parent: etree._Element, local_name: str) -> etree._Element | None:
        """Helper to safely fetch a child element by local name, ignoring namespaces."""
        if parent is None:
            return None
        for child in parent:
            if etree.QName(child.tag).localname == local_name:
                return child
        return None

