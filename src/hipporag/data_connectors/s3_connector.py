import boto3
import os
from typing import List, Dict, Optional
import io
import PyPDF2 # For PDF parsing
import docx # For DOCX parsing

# Configure logging
import logging
logger = logging.getLogger(__name__)

def get_documents_from_s3(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    region_name: Optional[str] = None,
    bucket_name: str = "",
    prefix: Optional[str] = ""
) -> List[str]:
    """
    Connects to AWS S3, fetches objects from the specified bucket and prefix,
    extracts text content from supported file types (.txt, .pdf, .docx), and returns a list of strings.

    Args:
        aws_access_key_id (Optional[str]): AWS access key ID. If None, boto3 will try to find credentials
                                            in environment variables or IAM roles.
        aws_secret_access_key (Optional[str]): AWS secret access key.
        region_name (Optional[str]): AWS region name.
        bucket_name (str): The name of the S3 bucket.
        prefix (Optional[str]): The prefix (folder path) within the bucket to fetch objects from.
                                 If empty, fetches from the root of the bucket.

    Returns:
        List[str]: A list of strings, where each string is the text content of a document.
                   Returns an empty list if the bucket is not found or no supported files are present.
    """
    if not bucket_name:
        logger.error("S3 bucket_name is required.")
        return []

    try:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        s3_resource = session.resource('s3')
        bucket = s3_resource.Bucket(bucket_name)
        
        # Check if bucket exists
        # s3_resource.meta.client.head_bucket(Bucket=bucket_name) # More robust check but requires ListBucket permission

    except Exception as e:
        logger.error(f"Failed to connect to S3 or access bucket '{bucket_name}': {e}")
        return []

    documents: List[str] = []
    logger.info(f"Fetching documents from S3 bucket: {bucket_name}, prefix: '{prefix or "(root)"}'")

    for obj in bucket.objects.filter(Prefix=prefix):
        file_key = obj.key
        try:
            # Skip directories/folders explicitly if their key ends with '/'
            if file_key.endswith('/'):
                logger.debug(f"Skipping directory-like object: {file_key}")
                continue

            file_content = obj.get()['Body'].read()
            text_content = ""

            if file_key.lower().endswith(".txt"):
                text_content = file_content.decode('utf-8')
                logger.debug(f"Successfully read text from: {file_key}")
            elif file_key.lower().endswith(".pdf"):
                with io.BytesIO(file_content) as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    for page_num in range(len(reader.pages)):
                        text_content += reader.pages[page_num].extract_text() or ""
                logger.debug(f"Successfully extracted text from PDF: {file_key}")
            elif file_key.lower().endswith(".docx"):
                with io.BytesIO(file_content) as docx_file:
                    doc = docx.Document(docx_file)
                    for para in doc.paragraphs:
                        text_content += para.text + "\n"
                logger.debug(f"Successfully extracted text from DOCX: {file_key}")
            else:
                logger.info(f"Skipping unsupported file type: {file_key}")
                continue
            
            if text_content.strip():
                documents.append(text_content.strip())
            else:
                logger.warning(f"Empty content extracted from: {file_key}")

        except Exception as e:
            logger.error(f"Error processing file {file_key} from S3: {e}")
    
    if not documents:
        logger.warning(f"No documents extracted from S3 path: s3://{bucket_name}/{prefix}")
    else:
        logger.info(f"Successfully extracted {len(documents)} documents from S3.")
            
    return documents

# Example Usage (optional, for testing purposes)
# if __name__ == '__main__':
#     # Configure your S3 details here or use environment variables
#     # Ensure your environment is configured for boto3 (e.g., ~/.aws/credentials)
#     # or pass credentials directly.
#     # Create a dummy file test.txt in a bucket/prefix for testing.
#     # s3_docs = get_documents_from_s3(
#     #     bucket_name="your-s3-bucket-name",
#     #     prefix="your-prefix/" 
#     # )
#     # for i, doc_content in enumerate(s3_docs):
#     #     print(f"--- Document {i+1} ---")
#     #     print(doc_content[:500] + "..." if len(doc_content) > 500 else doc_content)
#     pass
