import zipfile

from docx import Document
from lxml import etree
from loguru import logger

from frontend.clients.models import Review
from streamlit.runtime.uploaded_file_manager import UploadedFile


class UploadedFileProcessor:
    def __init__(
            self,
            uploaded_files: list[UploadedFile]
    ):
        super().__init__()
        self.uploaded_files = uploaded_files

    @staticmethod
    def get_numbering_prefix(numId, ilvl, numbering_root, level_counters):
        # numId is string, ilvl is int, level_counters is list
        namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        # Find <w:num> element with w:numId=numId
        num_xpath = f".//w:num[@w:numId='{numId}']"
        num_elem = numbering_root.find(num_xpath, namespaces=namespaces)
        if num_elem is not None:
            # Get the abstractNumId
            abstractNumId_elem = num_elem.find('.//w:abstractNumId', namespaces=namespaces)
            if abstractNumId_elem is not None:
                abstractNumId = abstractNumId_elem.get(
                    '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
                # Find <w:abstractNum> with w:abstractNumId=abstractNumId
                abstractNum_xpath = f".//w:abstractNum[@w:abstractNumId='{abstractNumId}']"
                abstractNum_elem = numbering_root.find(abstractNum_xpath, namespaces=namespaces)
                if abstractNum_elem is not None:
                    # Find <w:lvl> with w:ilvl=ilvl
                    lvl_xpath = f".//w:lvl[@w:ilvl='{ilvl}']"
                    lvl_elem = abstractNum_elem.find(lvl_xpath, namespaces=namespaces)
                    if lvl_elem is not None:
                        # Get <w:numFmt> and <w:lvlText>
                        numFmt_elem = lvl_elem.find('.//w:numFmt', namespaces=namespaces)
                        lvlText_elem = lvl_elem.find('.//w:lvlText', namespaces=namespaces)
                        if numFmt_elem is not None and lvlText_elem is not None:
                            numFmt = numFmt_elem.get(
                                '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
                            lvlText = lvlText_elem.get(
                                '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
                            # Now generate the numbering prefix
                            if numFmt == 'bullet':
                                # Return the bullet character (lvlText)
                                indent = '  ' * ilvl
                                return indent + lvlText + ' '
                            else:
                                # For numbered lists, replace placeholders with counters
                                numbering_prefix = lvlText
                                for i in range(len(level_counters)):
                                    numbering_prefix = numbering_prefix.replace(f'%{i + 1}', str(level_counters[i]))
                                indent = '  ' * ilvl
                                return indent + numbering_prefix + ' '
        # If we can't find the numbering format, return empty string
        return ''

    def process(self) -> list[Review]:
        reviews = self._extract_text()

        return reviews

    def _extract_text(self) -> list[Review]:
        processed_uploaded_files = []

        for idx, file in enumerate(self.uploaded_files):
            try:
                # Read the numbering.xml from the docm file
                with zipfile.ZipFile(file, 'r') as docx_zip:
                    # Read 'word/numbering.xml'
                    numbering_xml = docx_zip.read('word/numbering.xml')
                    # Parse numbering XML
                    numbering_root = etree.fromstring(numbering_xml)
                    # Read 'word/document.xml'
                    document_xml = docx_zip.read('word/document.xml')
                    # Parse the document XML
                    root = etree.fromstring(document_xml)

                # Define the namespaces
                namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

                form_fields = root.xpath('.//w:sdt', namespaces=namespaces)

                if form_fields:

                    tmp = {}

                    for sdt in form_fields:
                        # Extract the alias (name) of the form field
                        alias = sdt.xpath('.//w:alias/@w:val', namespaces=namespaces)
                        field_name = alias[0] if alias else 'N/A'

                        # Initialize field_value as an empty string
                        field_value = ''

                        numbering_counters = {}

                        # Get paragraphs within sdtContent
                        paragraphs = sdt.xpath('.//w:sdtContent//w:p', namespaces=namespaces)

                        for p in paragraphs:
                            # Check if paragraph is part of a list (has numbering properties)
                            numPr = p.find('.//w:numPr', namespaces=namespaces)
                            if numPr is not None:
                                # Get numId and ilvl
                                numId_elem = numPr.find('.//w:numId', namespaces=namespaces)
                                ilvl_elem = numPr.find('.//w:ilvl', namespaces=namespaces)
                                numId = numId_elem.get(
                                    '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val') if numId_elem is not None else None
                                ilvl = int(ilvl_elem.get(
                                    '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')) if ilvl_elem is not None else 0

                                # Initialize counters for numId if not present
                                if numId not in numbering_counters:
                                    numbering_counters[numId] = {}

                                # Reset counters for deeper levels
                                for level in range(ilvl + 1, 9):
                                    if level in numbering_counters[numId]:
                                        del numbering_counters[numId][level]

                                # Increment counter for current level
                                if ilvl not in numbering_counters[numId]:
                                    numbering_counters[numId][ilvl] = 1
                                else:
                                    numbering_counters[numId][ilvl] += 1

                                # Prepare the numbering sequence for levels up to current ilvl
                                level_counters = [numbering_counters[numId].get(lvl, 1) for lvl in range(ilvl + 1)]

                                # Get numbering prefix
                                numbering_prefix = self.get_numbering_prefix(numId, ilvl, numbering_root,
                                                                             level_counters)
                                field_value += numbering_prefix
                            else:
                                # Not a numbered or bulleted paragraph
                                pass
                            # Extract the text from the paragraph
                            texts = p.xpath('.//w:r//w:t/text()', namespaces=namespaces)
                            paragraph_text = ''.join(texts)
                            field_value += paragraph_text + '\n'

                        # Remove the last newline character
                        # field_value = field_value.rstrip('\n')

                        tmp[field_name] = field_value

                    review = Review(
                        reviewer=f"Reviewer {idx + 1}",
                        venue=tmp['Venue'],
                        summary=tmp['Summary'],
                        soundness=tmp['Soundness'],
                        presentation=tmp['Presentation'],
                        contribution=tmp['Contribution'],
                        strengths=tmp['Strengths'],
                        weaknesses=tmp['Weaknesses'],
                        questions=tmp['Questions'],
                        rating=tmp['Rating'],
                        confidence=tmp['Confidence']
                    )
                    processed_uploaded_files.append(review)
                else:
                    logger.warning("No form fields found in the document.")

            except Exception as e:
                logger.error(f"An error occurred: {e}")

        return processed_uploaded_files
