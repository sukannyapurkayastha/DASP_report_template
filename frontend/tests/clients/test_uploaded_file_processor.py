import pytest
from unittest.mock import MagicMock, patch
from lxml import etree

from frontend.clients.models import Review
from frontend.clients.uploaded_file_processor import UploadedFileProcessor


@pytest.fixture
def numbering_xml_bullet():
    """
    A minimal numbering.xml snippet containing a bullet list definition
    for testing get_numbering_prefix.
    """
    xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
        <w:num w:numId="5">
            <w:abstractNumId w:val="10"/>
        </w:num>
        <w:abstractNum w:abstractNumId="10">
            <w:lvl w:ilvl="0">
                <w:numFmt w:val="bullet"/>
                <w:lvlText w:val="•"/>
            </w:lvl>
        </w:abstractNum>
    </w:numbering>"""

    xml_bytes = xml.encode('utf-8')  # Ensure correct encoding of bullet
    return xml_bytes


@pytest.fixture
def numbering_xml_numbered():
    """
    A minimal numbering.xml snippet containing a numbered list definition
    for testing get_numbering_prefix.
    """
    xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
        <w:num w:numId="5">
            <w:abstractNumId w:val="10"/>
        </w:num>
        <w:abstractNum w:abstractNumId="10">
            <w:lvl w:ilvl="0">
                <w:numFmt w:val="decimal"/>
                <w:lvlText w:val="%1."/>
            </w:lvl>
            <w:lvl w:ilvl="1">
                <w:numFmt w:val="lowerLetter"/>
                <w:lvlText w:val="%2."/>
            </w:lvl>
        </w:abstractNum>
    </w:numbering>
    """

    xml_bytes = xml.encode('utf-8')
    return xml_bytes


def test_get_numbering_prefix_bullet(numbering_xml_bullet):
    """
    Test that get_numbering_prefix returns the bullet prefix correctly.
    """
    numbering_root = etree.fromstring(numbering_xml_bullet)
    # For bullet, we only need level_counters = [] or [1], it doesn't matter, since bullet formatting doesn't use them
    prefix = UploadedFileProcessor.get_numbering_prefix(
        numId="5",
        ilvl=0,
        numbering_root=numbering_root,
        level_counters=[]
    )
    # By default, indent is '  ' * ilvl (which is 0), so no indent, then the bullet character + space, i.e. "• "
    assert prefix == "• "


def test_get_numbering_prefix_numbered_top_level(numbering_xml_numbered):
    """
    Test that get_numbering_prefix returns a '1.' prefix for the top level (ilvl=0).
    """
    numbering_root = etree.fromstring(numbering_xml_numbered)
    prefix = UploadedFileProcessor.get_numbering_prefix(
        numId="5",
        ilvl=0,
        numbering_root=numbering_root,
        level_counters=[1]  # Assume numbering starts at 1
    )
    # For decimal format with lvlText="%1.", and counters = [1] => "1." no indent for ilvl=0
    assert prefix == "1. "


def test_get_numbering_prefix_numbered_second_level(numbering_xml_numbered):
    """
    Test that get_numbering_prefix returns something like '  a. ' for ilvl=1 with lowerLetter format.
    """
    numbering_root = etree.fromstring(numbering_xml_numbered)
    prefix = UploadedFileProcessor.get_numbering_prefix(
        numId="5",
        ilvl=1,
        numbering_root=numbering_root,
        level_counters=[1, 1]  # top level=1, second level=1
    )
    # For the second level (ilvl=1), there's an indent of '  ' * 1 => '  '
    # Then lvlText="%2." => it replaces %2 with level_counters[1] = 1 => "1."
    # So we'd expect "  1. "

    # For lettered ilvl=2 list we replace the letter with a number (simpler to handle)

    expected_prefix = "  1. "
    assert prefix == expected_prefix, f"Expected '{expected_prefix}', got '{prefix}'"


@pytest.fixture
def dummy_document_xml():
    """
    A minimal word/document.xml snippet with some <w:sdt> fields.
    """
    xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:sdt>
          <w:sdtPr>
            <w:alias w:val="Venue"/>
          </w:sdtPr>
          <w:sdtContent>
            <w:p>
              <w:r><w:t>Test Venue</w:t></w:r>
            </w:p>
          </w:sdtContent>
        </w:sdt>
        <w:sdt>
          <w:sdtPr>
            <w:alias w:val="Summary"/>
          </w:sdtPr>
          <w:sdtContent>
            <w:p>
              <w:r><w:t>Test Summary</w:t></w:r>
            </w:p>
          </w:sdtContent>
        </w:sdt>
        <w:sdt>
          <w:sdtPr>
            <w:alias w:val="Soundness"/>
          </w:sdtPr>
          <w:sdtContent>
            <w:p>
              <w:numPr>
                <w:ilvl w:val="0"/>
                <w:numId w:val="5"/>
              </w:numPr>
              <w:r><w:t>Point 1</w:t></w:r>
            </w:p>
          </w:sdtContent>
        </w:sdt>
        <w:sdt>
          <w:sdtPr>
            <w:alias w:val="Presentation"/>
          </w:sdtPr>
          <w:sdtContent>
            <w:p>
              <w:r><w:t>Good</w:t></w:r>
            </w:p>
          </w:sdtContent>
        </w:sdt>
        <w:sdt>
          <w:sdtPr>
            <w:alias w:val="Contribution"/>
          </w:sdtPr>
          <w:sdtContent>
            <w:p>
              <w:r><w:t>High</w:t></w:r>
            </w:p>
          </w:sdtContent>
        </w:sdt>
        <w:sdt>
          <w:sdtPr>
            <w:alias w:val="Strengths"/>
          </w:sdtPr>
          <w:sdtContent>
            <w:p>
              <w:r><w:t>Clear</w:t></w:r>
            </w:p>
          </w:sdtContent>
        </w:sdt>
        <w:sdt>
          <w:sdtPr>
            <w:alias w:val="Weaknesses"/>
          </w:sdtPr>
          <w:sdtContent>
            <w:p>
              <w:r><w:t>None</w:t></w:r>
            </w:p>
          </w:sdtContent>
        </w:sdt>
        <w:sdt>
          <w:sdtPr>
            <w:alias w:val="Questions"/>
          </w:sdtPr>
          <w:sdtContent>
            <w:p>
              <w:r><w:t>Any clarifications?</w:t></w:r>
            </w:p>
          </w:sdtContent>
        </w:sdt>
        <w:sdt>
          <w:sdtPr>
            <w:alias w:val="Rating"/>
          </w:sdtPr>
          <w:sdtContent>
            <w:p>
              <w:r><w:t>5</w:t></w:r>
            </w:p>
          </w:sdtContent>
        </w:sdt>
        <w:sdt>
          <w:sdtPr>
            <w:alias w:val="Confidence"/>
          </w:sdtPr>
          <w:sdtContent>
            <w:p>
              <w:r><w:t>High</w:t></w:r>
            </w:p>
          </w:sdtContent>
        </w:sdt>
      </w:body>
    </w:document>
    """
    xml_bytes = xml.encode("utf-8")
    return xml_bytes


@pytest.fixture
def mock_uploaded_file():
    """
    Create a mock for streamlit's UploadedFile object. We just need a 'name' attribute and to behave like a file in
    zipfile.ZipFile.
    """
    file_mock = MagicMock()
    file_mock.name = "dummy.docm"
    return file_mock


@patch("zipfile.ZipFile")
def test_process_single_file(
        mock_zipfile_class,
        mock_uploaded_file,
        numbering_xml_numbered,
        dummy_document_xml
):
    """
    Test that UploadedFileProcessor.process() successfully reads from a mocked docx/docm and returns a list with one
    Review object.
    """
    # 1. Mock the context manager returned by zipfile.ZipFile(file, 'r')
    mock_zipfile_instance = MagicMock()

    # 2. read() should return numbering_xml for 'word/numbering.xml', and document_xml for 'word/document.xml'.
    def side_effect(filename):
        if filename == "word/numbering.xml":
            return numbering_xml_numbered
        elif filename == "word/document.xml":
            return dummy_document_xml
        else:
            raise FileNotFoundError(f"Mock doesn't have {filename}")

    mock_zipfile_instance.read.side_effect = side_effect

    # 3. Make the mock_zipfile_class return our mock_zipfile_instance
    mock_zipfile_class.return_value.__enter__.return_value = mock_zipfile_instance

    # 4. Build the processor with a single "uploaded file"
    processor = UploadedFileProcessor(uploaded_files=[mock_uploaded_file])
    reviews = processor.process()

    # 5. Assertions
    assert len(reviews) == 1, "Expected exactly one Review object"

    review = reviews[0]
    assert review.reviewer == "Reviewer 1"
    assert review.venue == "Test Venue\n"
    assert review.summary == "Test Summary\n"
    assert "Point 1" in review.soundness  # Should include the numbering prefix
    assert review.presentation == "Good\n"
    assert review.contribution == "High\n"
    assert review.strengths == "Clear\n"
    assert review.weaknesses == "None\n"
    assert review.questions == "Any clarifications?\n"
    assert review.rating == "5\n"
    assert review.confidence == "High\n"

    # Check that we actually used zipfile on our mock
    mock_zipfile_class.assert_called_once_with(mock_uploaded_file, "r")


@patch("zipfile.ZipFile")
def test_process_no_form_fields(
        mock_zipfile_class,
        mock_uploaded_file
):
    """
    Test that if the document XML has no <w:sdt> elements, we get an empty list and a warning is logged (no exception).
    """
    empty_doc_xml = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p>
          <w:r><w:t>No form fields here!</w:t></w:r>
        </w:p>
      </w:body>
    </w:document>
    """

    # Mock the zipfile instance
    mock_zipfile_instance = MagicMock()
    mock_zipfile_instance.read.side_effect = lambda filename: (
        b"<w:numbering xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main' />"
        if filename == "word/numbering.xml"
        else empty_doc_xml
    )
    mock_zipfile_class.return_value.__enter__.return_value = mock_zipfile_instance

    processor = UploadedFileProcessor(uploaded_files=[mock_uploaded_file])
    reviews = processor.process()

    # We expect no reviews because there's no <w:sdt> present
    assert len(reviews) == 0


@patch("zipfile.ZipFile")
def test_process_with_exception(
        mock_zipfile_class,
        mock_uploaded_file
):
    """
    Test that if an exception is raised while reading the file, we log an error and continue gracefully (returning no
    reviews).
    """
    # Force a read() to raise an exception
    mock_zipfile_instance = MagicMock()
    mock_zipfile_instance.read.side_effect = Exception("Mocked read error")
    mock_zipfile_class.return_value.__enter__.return_value = mock_zipfile_instance

    processor = UploadedFileProcessor(uploaded_files=[mock_uploaded_file])
    reviews = processor.process()

    # Because an exception is raised, we expect zero reviews,
    # but the method should handle it gracefully without re-raising.
    assert len(reviews) == 0
