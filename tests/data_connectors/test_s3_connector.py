import pytest
from unittest.mock import patch, MagicMock
from src.hipporag.data_connectors.s3_connector import get_documents_from_s3
import io
import PyPDF2
import docx

@pytest.fixture
def mock_s3_resource():
    with patch('boto3.Session') as mock_session:
        mock_s3 = MagicMock()
        mock_bucket = MagicMock()
        mock_s3.Bucket.return_value = mock_bucket
        
        # Simulate head_bucket success (bucket exists)
        # mock_s3.meta.client.head_bucket.return_value = {} # No longer used in current s3_connector

        mock_session.return_value.resource.return_value = mock_s3
        yield mock_s3, mock_bucket # yield both for easier mocking of objects

def test_get_documents_from_s3_txt(mock_s3_resource):
    mock_s3, mock_bucket = mock_s3_resource
    
    txt_object_mock = MagicMock()
    txt_object_mock.key = "test.txt"
    txt_object_mock.get.return_value = {'Body': io.BytesIO(b"Hello S3 TXT")}
    
    mock_bucket.objects.filter.return_value = [txt_object_mock]

    docs = get_documents_from_s3(bucket_name="test-bucket", prefix="test/")
    assert len(docs) == 1
    assert "Hello S3 TXT" in docs[0]

def test_get_documents_from_s3_pdf(mock_s3_resource):
    mock_s3, mock_bucket = mock_s3_resource

    # Create a dummy PDF in memory
    pdf_writer = PyPDF2.PdfWriter()
    pdf_writer.add_blank_page(width=612, height=792) # Standard letter size
    # Add text to the PDF - PyPDF2 writer doesn't directly add text easily.
    # For a unit test, mocking extract_text is more reliable.
    # However, we'll create a simple PDF and mock the reader if needed.
    pdf_bytes_io = io.BytesIO()
    pdf_writer.write(pdf_bytes_io)
    pdf_bytes_io.seek(0)

    pdf_object_mock = MagicMock()
    pdf_object_mock.key = "test.pdf"
    pdf_object_mock.get.return_value = {'Body': pdf_bytes_io}
    
    mock_bucket.objects.filter.return_value = [pdf_object_mock]

    # Mock PyPDF2.PdfReader
    with patch('PyPDF2.PdfReader') as mock_pdf_reader_constructor:
        mock_pdf_reader_instance = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Hello S3 PDF"
        mock_pdf_reader_instance.pages = [mock_page]
        mock_pdf_reader_constructor.return_value = mock_pdf_reader_instance
        
        docs = get_documents_from_s3(bucket_name="test-bucket", prefix="test/")
        assert len(docs) == 1
        assert "Hello S3 PDF" in docs[0]

def test_get_documents_from_s3_docx(mock_s3_resource):
    mock_s3, mock_bucket = mock_s3_resource

    # Create a dummy DOCX in memory
    doc = docx.Document()
    doc.add_paragraph("Hello S3 DOCX")
    docx_bytes_io = io.BytesIO()
    doc.save(docx_bytes_io)
    docx_bytes_io.seek(0)

    docx_object_mock = MagicMock()
    docx_object_mock.key = "test.docx"
    docx_object_mock.get.return_value = {'Body': docx_bytes_io}

    mock_bucket.objects.filter.return_value = [docx_object_mock]
    
    docs = get_documents_from_s3(bucket_name="test-bucket", prefix="test/")
    assert len(docs) == 1
    assert "Hello S3 DOCX" in docs[0]

def test_get_documents_from_s3_unsupported_type(mock_s3_resource):
    mock_s3, mock_bucket = mock_s3_resource
    
    img_object_mock = MagicMock()
    img_object_mock.key = "test.jpg"
    img_object_mock.get.return_value = {'Body': io.BytesIO(b"")} # Content doesn't matter

    mock_bucket.objects.filter.return_value = [img_object_mock]
    docs = get_documents_from_s3(bucket_name="test-bucket")
    assert len(docs) == 0

def test_get_documents_from_s3_empty_bucket(mock_s3_resource):
    mock_s3, mock_bucket = mock_s3_resource
    mock_bucket.objects.filter.return_value = []
    docs = get_documents_from_s3(bucket_name="test-bucket")
    assert len(docs) == 0

def test_get_documents_from_s3_connection_error(mock_s3_resource):
    mock_s3, _ = mock_s3_resource # Unpack but don't use mock_bucket here
    mock_s3.Bucket.side_effect = Exception("AWS Connection Error")
    docs = get_documents_from_s3(bucket_name="test-bucket")
    assert len(docs) == 0
    # Add assertion for logging if logger is captured

def test_get_documents_from_s3_no_bucket_name(mock_s3_resource):
    docs = get_documents_from_s3(bucket_name="") # No bucket name
    assert len(docs) == 0

def test_get_documents_from_s3_directory_object(mock_s3_resource):
    mock_s3, mock_bucket = mock_s3_resource
    
    dir_object_mock = MagicMock()
    dir_object_mock.key = "myfolder/" # Ends with a slash
    # No .get() method will be called on this if logic is correct
    
    txt_object_mock = MagicMock()
    txt_object_mock.key = "myfolder/test.txt"
    txt_object_mock.get.return_value = {'Body': io.BytesIO(b"Hello from subfolder")}

    mock_bucket.objects.filter.return_value = [dir_object_mock, txt_object_mock]

    docs = get_documents_from_s3(bucket_name="test-bucket", prefix="myfolder/")
    assert len(docs) == 1
    assert "Hello from subfolder" in docs[0]
