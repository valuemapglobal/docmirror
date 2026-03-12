import pytest
from lxml import etree
from docmirror.adapters.office.omml_extractor import OMMLExtractor

def test_basic_conversion():
    xml = """
    <m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
      <m:r><m:t>E=m</m:t></m:r>
      <m:sSup>
        <m:e><m:r><m:t>c</m:t></m:r></m:e>
        <m:sup><m:r><m:t>2</m:t></m:r></m:sup>
      </m:sSup>
    </m:oMath>
    """
    node = etree.fromstring(xml.encode("utf-8"))
    latex = OMMLExtractor.convert_element(node)
    assert latex == "E=mc^{2}"

def test_fraction():
    xml = """
    <m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
        <m:f>
            <m:num><m:r><m:t>a</m:t></m:r></m:num>
            <m:den><m:r><m:t>b</m:t></m:r></m:den>
        </m:f>
    </m:oMath>
    """
    node = etree.fromstring(xml.encode("utf-8"))
    latex = OMMLExtractor.convert_element(node)
    assert latex == "\\frac{a}{b}"

def test_subscript():
    xml = """
    <m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
        <m:sSub>
            <m:e><m:r><m:t>H</m:t></m:r></m:e>
            <m:sub><m:r><m:t>2</m:t></m:r></m:sub>
        </m:sSub>
        <m:r><m:t>O</m:t></m:r>
    </m:oMath>
    """
    node = etree.fromstring(xml.encode("utf-8"))
    latex = OMMLExtractor.convert_element(node)
    assert latex == "H_{2}O"
