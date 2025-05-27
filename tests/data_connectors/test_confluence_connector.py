import pytest
from unittest.mock import patch, MagicMock
from src.hipporag.data_connectors.confluence_connector import get_documents_from_confluence
import io
import PyPDF2 # Required for the main code's PDF processing

@pytest.fixture
def mock_confluence_client():
    with patch('src.hipporag.data_connectors.confluence_connector.Confluence') as mock_confluence_constructor:
        mock_instance = MagicMock()
        mock_confluence_constructor.return_value = mock_instance
        yield mock_instance

def test_get_documents_from_confluence_by_space(mock_confluence_client):
    page1 = {
        'id': '123', 'title': 'Test Page 1', 
        'body': {'storage': {'value': '<p>Hello Confluence Page 1</p>'}},
        '_links': {} # Ensure _links is present
    }
    mock_confluence_client.get_all_pages_from_space.return_value = [page1]
    mock_confluence_client.get_attachments_from_content.return_value = {'results': []} # No attachments

    docs = get_documents_from_confluence(
        confluence_url="https://fake.confluence.com", 
        username="user", 
        api_token="token", 
        space_key="TESTSPACE"
    )
    assert len(docs) == 1
    assert "Page Title: Test Page 1" in docs[0]
    assert "Hello Confluence Page 1" in docs[0]
    mock_confluence_client.get_all_pages_from_space.assert_called_once()

def test_get_documents_from_confluence_by_cql(mock_confluence_client):
    page2 = {
        'id': '456', 'title': 'Test Page 2 CQL',
        'body': {'storage': {'value': 'Content for CQL page.'}},
        '_links': {}
    }
    mock_confluence_client.cql.return_value = {'results': [page2]}
    mock_confluence_client.get_attachments_from_content.return_value = {'results': []}

    docs = get_documents_from_confluence(
        confluence_url="https://fake.confluence.com", 
        username="user", 
        api_token="token", 
        cql="type=page"
    )
    assert len(docs) == 1
    assert "Page Title: Test Page 2 CQL" in docs[0]
    assert "Content for CQL page." in docs[0]
    mock_confluence_client.cql.assert_called_once()

def test_get_documents_from_confluence_with_txt_attachment(mock_confluence_client):
    page_with_attach = {
        'id': '789', 'title': 'Page With Attachment',
        'body': {'storage': {'value': 'Main content.'}},
        '_links': {}
    }
    attachment1 = {
        'id': 'att1', 'title': 'attach.txt', 
        'metadata': {'mediaType': 'text/plain'}, 
        '_links': {'download': '/download/att1.txt'}
    }
    mock_confluence_client.get_all_pages_from_space.return_value = [page_with_attach]
    mock_confluence_client.get_attachments_from_content.return_value = {'results': [attachment1]}
    # Mock the download itself
    mock_confluence_client.get_attachment_by_id.return_value = b"Hello Attachment TXT"

    docs = get_documents_from_confluence(
        confluence_url="https://fake.confluence.com", 
        username="user", 
        api_token="token", 
        space_key="TESTSPACE",
        include_attachments=True
    )
    assert len(docs) == 2 # Page + attachment
    assert "Attachment from Page 'Page With Attachment': attach.txt" in docs[1]
    assert "Hello Attachment TXT" in docs[1]
    mock_confluence_client.get_attachment_by_id.assert_called_once_with(
        content_id='789', attachment_filename='attach.txt', download=True
    )

def test_get_documents_from_confluence_with_pdf_attachment(mock_confluence_client):
    page_with_pdf = {
        'id': '101', 'title': 'Page With PDF',
        'body': {'storage': {'value': 'Main PDF content.'}},
        '_links': {}
    }
    pdf_attachment = {
        'id': 'pdf_att1', 'title': 'document.pdf',
        'metadata': {'mediaType': 'application/pdf'},
        '_links': {'download': '/download/document.pdf'}
    }
    mock_confluence_client.get_all_pages_from_space.return_value = [page_with_pdf]
    mock_confluence_client.get_attachments_from_content.return_value = {'results': [pdf_attachment]}
    
    # Create a dummy PDF bytes for download mock
    pdf_writer = PyPDF2.PdfWriter()
    pdf_writer.add_blank_page(width=612, height=792)
    pdf_bytes_io = io.BytesIO()
    pdf_writer.write(pdf_bytes_io)
    pdf_bytes_io.seek(0)
    mock_confluence_client.get_attachment_by_id.return_value = pdf_bytes_io.getvalue()

    # Mock PyPDF2.PdfReader for attachment processing
    with patch('PyPDF2.PdfReader') as mock_pdf_reader_constructor:
        mock_pdf_reader_instance = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Hello Attachment PDF"
        mock_pdf_reader_instance.pages = [mock_page]
        mock_pdf_reader_constructor.return_value = mock_pdf_reader_instance

        docs = get_documents_from_confluence(
            confluence_url="https://fake.confluence.com",
            username="user", api_token="token", space_key="TESTSPACE",
            include_attachments=True
        )
        assert len(docs) == 2
        assert "Attachment from Page 'Page With PDF': document.pdf" in docs[1]
        assert "Hello Attachment PDF" in docs[1]

def test_get_documents_from_confluence_no_url(mock_confluence_client):
    docs = get_documents_from_confluence(confluence_url="", username="user", api_token="token", space_key="TEST")
    assert len(docs) == 0

def test_get_documents_from_confluence_no_space_or_cql(mock_confluence_client):
    docs = get_documents_from_confluence(confluence_url="https://fake.confluence.com", username="user", api_token="token")
    assert len(docs) == 0
    
def test_get_documents_from_confluence_connection_error(mock_confluence_client):
    # Simulate an error when trying to make the initial connection or a call like get_all_pages_from_space
    mock_confluence_client.get_all_pages_from_space.side_effect = Exception("Confluence Connection Error")
    # OR mock_confluence_constructor.side_effect if connection fails at instantiation
    # with patch('src.hipporag.data_connectors.confluence_connector.Confluence', side_effect=Exception("Init fail")):

    docs = get_documents_from_confluence(
        confluence_url="https://fake.confluence.com", 
        username="user", 
        api_token="token", 
        space_key="TESTSPACE"
    )
    assert len(docs) == 0

def test_get_documents_from_confluence_no_credentials(mock_confluence_client, monkeypatch):
    # Ensure env vars are not set for this test
    monkeypatch.delenv("CONFLUENCE_USERNAME", raising=False)
    monkeypatch.delenv("CONFLUENCE_API_TOKEN", raising=False)
    
    docs = get_documents_from_confluence(
        confluence_url="https://fake.confluence.com", 
        username=None, # Explicitly None
        api_token=None, # Explicitly None
        space_key="TESTSPACE"
    )
    assert len(docs) == 0
    # The Confluence client mock shouldn't even be called if credentials checks fail first
    mock_confluence_client.get_all_pages_from_space.assert_not_called()

def test_get_documents_from_confluence_unsupported_attachment(mock_confluence_client):
    page_with_unsupported_attach = {
        'id': 'unsupported1', 'title': 'Page With Other Attachment',
        'body': {'storage': {'value': 'Some content.'}},
        '_links': {}
    }
    unsupported_attachment = {
        'id': 'att_unsupported', 'title': 'image.jpg', 
        'metadata': {'mediaType': 'image/jpeg'}, 
        '_links': {'download': '/download/image.jpg'}
    }
    mock_confluence_client.get_all_pages_from_space.return_value = [page_with_unsupported_attach]
    mock_confluence_client.get_attachments_from_content.return_value = {'results': [unsupported_attachment]}
    # We don't need to mock get_attachment_by_id as it shouldn't be called for unsupported type

    docs = get_documents_from_confluence(
        confluence_url="https://fake.confluence.com", 
        username="user", 
        api_token="token", 
        space_key="TESTSPACE",
        include_attachments=True
    )
    assert len(docs) == 1 # Only the page content
    assert "Page Title: Page With Other Attachment" in docs[0]
    mock_confluence_client.get_attachment_by_id.assert_not_called()

def test_get_documents_from_confluence_empty_page_body(mock_confluence_client):
    empty_body_page = {
        'id': 'empty1', 'title': 'Empty Body Page',
        'body': {'storage': {'value': ''}}, # Empty body
        '_links': {}
    }
    mock_confluence_client.get_all_pages_from_space.return_value = [empty_body_page]
    mock_confluence_client.get_attachments_from_content.return_value = {'results': []}

    docs = get_documents_from_confluence(
        confluence_url="https://fake.confluence.com", username="user", api_token="token", space_key="TEST"
    )
    # Depending on how strictly "empty" is handled, it might be an empty string or not included.
    # Current connector code adds "Page Title: ...\n\n" then the content. If content is empty, it's still a doc.
    # Let's assume it still produces a document with just the title if the body is empty.
    assert len(docs) == 1 
    assert "Page Title: Empty Body Page" in docs[0]
    assert docs[0].strip().endswith("Page Title: Empty Body Page") # Check that no body content follows significantly

def test_get_documents_from_confluence_no_pages_found(mock_confluence_client):
    mock_confluence_client.get_all_pages_from_space.return_value = [] # No pages
    mock_confluence_client.cql.return_value = {'results': []} # No pages for CQL either

    docs_space = get_documents_from_confluence(
        confluence_url="https://fake.confluence.com", username="user", api_token="token", space_key="TEST_EMPTY"
    )
    assert len(docs_space) == 0

    docs_cql = get_documents_from_confluence(
        confluence_url="https://fake.confluence.com", username="user", api_token="token", cql="label=nonexistent"
    )
    assert len(docs_cql) == 0

# Add PyPDF2 to requirements if not already implied by the main code.
# It's used in the connector, so it should be.
