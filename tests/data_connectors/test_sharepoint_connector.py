import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from src.hipporag.data_connectors.sharepoint_connector import get_documents_from_sharepoint
from office365.sharepoint.files.file import File # For mocking File.open_binary
import io
import PyPDF2 # For PDF and DOCX testing, as they are used by the connector
import docx

@pytest.fixture
def mock_sharepoint_context():
    with patch('src.hipporag.data_connectors.sharepoint_connector.ClientContext') as mock_context_constructor:
        mock_ctx = MagicMock()
        
        # Mock web properties for server_relative_url
        mock_web = MagicMock()
        # Use PropertyMock for attributes like server_relative_url
        # Re-assigning to 'type(mock_web).server_relative_url' is the way to mock properties on a MagicMock instance
        type(mock_web).server_relative_url = PropertyMock(return_value="/sites/MySite")
        mock_ctx.web.get.return_value = mock_web # mock_ctx.web.get() returns the mock_web

        mock_folder = MagicMock()
        # Mocking properties of 'properties' attribute if 'properties' itself is an object
        # If properties is a dict, then it's mock_folder.properties = {"Name": "RootFolder", "ServerRelativeUrl": ...}
        # Assuming properties is an object that will have attributes 'Name' and 'ServerRelativeUrl'
        type(mock_folder.properties).name = PropertyMock(return_value="RootFolder") 
        type(mock_folder.properties).server_relative_url = PropertyMock(return_value="/sites/MySite/Shared Documents")


        mock_ctx.web.get_folder_by_server_relative_url.return_value = mock_folder
        # Ensure with_credentials returns the mock_ctx for chained calls
        mock_context_constructor.return_value.with_credentials.return_value = mock_ctx 
        yield mock_ctx, mock_folder


# Need to patch 'File.open_binary' correctly within the module it's USED.
# The SharePoint connector calls office365.sharepoint.files.file.File.open_binary
@patch('src.hipporag.data_connectors.sharepoint_connector.File.open_binary')
def test_get_documents_from_sharepoint_txt(mock_open_binary, mock_sharepoint_context):
    mock_ctx, mock_folder = mock_sharepoint_context

    mock_file_item = MagicMock()
    # Mocking properties of 'properties' attribute of mock_file_item
    type(mock_file_item.properties).name = PropertyMock(return_value="test.txt")
    type(mock_file_item.properties).server_relative_url = PropertyMock(return_value="/sites/MySite/Shared Documents/test.txt")
    
    mock_folder.files = [mock_file_item] # Folder contains this one file
    mock_folder.folders = [] # No subfolders for this test

    # Mock the return of File.open_binary
    # It should return an object that has a 'content' attribute which is bytes
    mock_file_response = MagicMock()
    mock_file_response.content = b"Hello SharePoint TXT"
    mock_open_binary.return_value = mock_file_response


    docs = get_documents_from_sharepoint(
        sharepoint_url="https://fake.sharepoint.com/sites/MySite",
        username="user",
        password="password",
        document_library_name="Shared Documents"
    )
    assert len(docs) == 1
    assert "Hello SharePoint TXT" in docs[0]
    assert "SharePoint File: test.txt" in docs[0]
    mock_open_binary.assert_called_once_with(mock_ctx, "/sites/MySite/Shared Documents/test.txt")


@patch('src.hipporag.data_connectors.sharepoint_connector.File.open_binary')
@patch('PyPDF2.PdfReader') # Mock PdfReader used by the connector
def test_get_documents_from_sharepoint_pdf(mock_pdf_reader_constructor, mock_open_binary, mock_sharepoint_context):
    mock_ctx, mock_folder = mock_sharepoint_context

    mock_file_item = MagicMock()
    type(mock_file_item.properties).name = PropertyMock(return_value="test.pdf")
    type(mock_file_item.properties).server_relative_url = PropertyMock(return_value="/sites/MySite/Shared Documents/test.pdf")
    
    mock_folder.files = [mock_file_item]
    mock_folder.folders = []

    # Dummy PDF bytes (content doesn't matter as PdfReader is mocked)
    mock_file_response = MagicMock()
    mock_file_response.content = b"dummy pdf bytes"
    mock_open_binary.return_value = mock_file_response 

    # Configure the mock PdfReader instance
    mock_pdf_reader_instance = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Hello SharePoint PDF"
    mock_pdf_reader_instance.pages = [mock_page] # pages should be a list of page mocks
    mock_pdf_reader_constructor.return_value = mock_pdf_reader_instance

    docs = get_documents_from_sharepoint(
        sharepoint_url="https://fake.sharepoint.com/sites/MySite",
        username="user", password="password", document_library_name="Shared Documents"
    )
    assert len(docs) == 1
    assert "Hello SharePoint PDF" in docs[0]

@patch('src.hipporag.data_connectors.sharepoint_connector.File.open_binary')
@patch('docx.Document') # Mock Document used by the connector
def test_get_documents_from_sharepoint_docx(mock_docx_constructor, mock_open_binary, mock_sharepoint_context):
    mock_ctx, mock_folder = mock_sharepoint_context

    mock_file_item = MagicMock()
    type(mock_file_item.properties).name = PropertyMock(return_value="test.docx")
    type(mock_file_item.properties).server_relative_url = PropertyMock(return_value="/sites/MySite/Shared Documents/test.docx")

    mock_folder.files = [mock_file_item]
    mock_folder.folders = []
    
    mock_file_response = MagicMock()
    mock_file_response.content = b"dummy docx bytes"
    mock_open_binary.return_value = mock_file_response

    # Configure the mock Document instance
    mock_docx_instance = MagicMock()
    mock_para = MagicMock()
    mock_para.text = "Hello SharePoint DOCX"
    mock_docx_instance.paragraphs = [mock_para]
    mock_docx_constructor.return_value = mock_docx_instance
    
    docs = get_documents_from_sharepoint(
        sharepoint_url="https://fake.sharepoint.com/sites/MySite",
        username="user", password="password", document_library_name="Shared Documents"
    )
    assert len(docs) == 1
    assert "Hello SharePoint DOCX" in docs[0]

def test_get_documents_from_sharepoint_no_url(mock_sharepoint_context):
    # No need for mock_ctx or mock_folder if the function exits early
    docs = get_documents_from_sharepoint(sharepoint_url="", username="user", password="password")
    assert len(docs) == 0

def test_get_documents_from_sharepoint_auth_error(mock_sharepoint_context):
    # mock_sharepoint_context already patches ClientContext constructor
    # We need to get the original constructor mock to change its side_effect or return_value
    mock_ctx_constructor = mock_sharepoint_context[0].parent 
    
    # Simulate auth failure by having with_credentials raise an exception
    # ClientContext(...).with_credentials(...) is the chain.
    # mock_ctx_constructor.return_value is the ClientContext instance mock.
    # So, mock_ctx_constructor.return_value.with_credentials is what we need to mock.
    mock_ctx_constructor.return_value.with_credentials.side_effect = Exception("SharePoint Auth Error")
    
    docs = get_documents_from_sharepoint(
        sharepoint_url="https://fake.sharepoint.com/sites/MySite",
        username="user", password="password"
    )
    assert len(docs) == 0
    
@patch('src.hipporag.data_connectors.sharepoint_connector.File.open_binary')
def test_get_documents_from_sharepoint_recursive(mock_open_binary, mock_sharepoint_context):
    mock_ctx, mock_root_folder = mock_sharepoint_context

    # Root folder file
    mock_file_item_root = MagicMock()
    type(mock_file_item_root.properties).name = PropertyMock(return_value="root.txt")
    type(mock_file_item_root.properties).server_relative_url = PropertyMock(return_value="/sites/MySite/Shared Documents/root.txt")
    
    # Subfolder and its file
    mock_sub_folder = MagicMock()
    type(mock_sub_folder.properties).name = PropertyMock(return_value="SubFolder1")
    type(mock_sub_folder.properties).server_relative_url = PropertyMock(return_value="/sites/MySite/Shared Documents/SubFolder1")
    
    mock_file_item_sub = MagicMock()
    type(mock_file_item_sub.properties).name = PropertyMock(return_value="sub.txt")
    type(mock_file_item_sub.properties).server_relative_url = PropertyMock(return_value="/sites/MySite/Shared Documents/SubFolder1/sub.txt")

    mock_sub_folder.files = [mock_file_item_sub]
    mock_sub_folder.folders = [] # No further recursion for this subfolder

    mock_root_folder.files = [mock_file_item_root]
    mock_root_folder.folders = [mock_sub_folder] # Root folder contains one subfolder

    # Mock File.open_binary to return different content based on URL
    def open_binary_side_effect(ctx, server_relative_url):
        mock_response = MagicMock()
        if server_relative_url.endswith("root.txt"):
            mock_response.content = b"Root TXT content"
        elif server_relative_url.endswith("sub.txt"):
            mock_response.content = b"Subfolder TXT content"
        else:
            mock_response.content = b"" # Should not happen in this test
        return mock_response

    mock_open_binary.side_effect = open_binary_side_effect

    docs = get_documents_from_sharepoint(
        sharepoint_url="https://fake.sharepoint.com/sites/MySite",
        username="user", password="password", document_library_name="Shared Documents",
        recursive=True
    )
    assert len(docs) == 2
    doc_contents_combined = " ".join(docs) # Combine for easier checking
    assert "Root TXT content" in doc_contents_combined
    assert "Subfolder TXT content" in doc_contents_combined

def test_get_documents_from_sharepoint_no_creds(mock_sharepoint_context, monkeypatch):
    # Ensure env vars are not set for this test
    monkeypatch.delenv("SHAREPOINT_USERNAME", raising=False)
    monkeypatch.delenv("SHAREPOINT_PASSWORD", raising=False)
    
    docs = get_documents_from_sharepoint(
        sharepoint_url="https://fake.sharepoint.com/sites/MySite",
        username=None, # Explicitly None
        password=None, # Explicitly None
        document_library_name="Shared Documents"
    )
    assert len(docs) == 0
    # ClientContext should not have been called if creds are missing
    mock_ctx_constructor = mock_sharepoint_context[0].parent
    mock_ctx_constructor.assert_not_called()

@patch('src.hipporag.data_connectors.sharepoint_connector.File.open_binary')
def test_get_documents_from_sharepoint_unsupported_file(mock_open_binary, mock_sharepoint_context):
    mock_ctx, mock_folder = mock_sharepoint_context
    
    mock_file_item_unsupported = MagicMock()
    type(mock_file_item_unsupported.properties).name = PropertyMock(return_value="image.jpg")
    type(mock_file_item_unsupported.properties).server_relative_url = PropertyMock(return_value="/sites/MySite/Shared Documents/image.jpg")

    mock_folder.files = [mock_file_item_unsupported]
    mock_folder.folders = []

    # open_binary would still be called, but its content won't be processed for text
    mock_file_response = MagicMock()
    mock_file_response.content = b"jpeg image data"
    mock_open_binary.return_value = mock_file_response

    docs = get_documents_from_sharepoint(
        sharepoint_url="https://fake.sharepoint.com/sites/MySite",
        username="user", password="password", document_library_name="Shared Documents"
    )
    assert len(docs) == 0
    mock_open_binary.assert_called_once_with(mock_ctx, "/sites/MySite/Shared Documents/image.jpg")
    
# Note: PyPDF2 and python-docx are used by the connector, so they are part of its dependencies.
# Test dependencies like pytest and pytest-mock are for the test environment.
