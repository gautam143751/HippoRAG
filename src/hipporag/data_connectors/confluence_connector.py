from atlassian import Confluence
from typing import List, Optional, Dict
import os

# Configure logging
import logging
logger = logging.getLogger(__name__)

def get_documents_from_confluence(
    confluence_url: str,
    username: Optional[str] = None, # Or use api_token with cloud
    api_token: Optional[str] = None, # Or use password with on-prem
    space_key: Optional[str] = None,
    cql: Optional[str] = None,
    limit: int = 50,
    include_attachments: bool = False, # Basic attachment text extraction (txt, pdf)
    attachment_limit_per_page: int = 10
) -> List[str]:
    """
    Connects to Confluence, fetches pages from the specified space or using CQL,
    extracts text content, and optionally text from attachments.

    Args:
        confluence_url (str): The URL of the Confluence instance (e.g., "https://yourcompany.atlassian.net/wiki").
        username (Optional[str]): Username for Confluence authentication.
                                   For Confluence Cloud, this is typically the email address.
        api_token (Optional[str]): API token for Confluence authentication.
                                   For Confluence Cloud, generate this from account settings.
                                   For on-prem, this might be a password or personal access token.
        space_key (Optional[str]): The key of the Confluence space to fetch pages from.
                                   If None, 'cql' must be provided.
        cql (Optional[str]): Confluence Query Language (CQL) string to search for pages.
                             Example: 'label="mydoclabel" and type=page'.
                             If provided, 'space_key' is ignored for page fetching.
        limit (int): Maximum number of pages to retrieve.
        include_attachments (bool): Whether to attempt to download and extract text from attachments.
        attachment_limit_per_page (int): Max number of attachments to process per page.

    Returns:
        List[str]: A list of strings, where each string is the text content of a page or attachment.
                   Returns an empty list on failure or if no content is found.
    """
    if not confluence_url:
        logger.error("Confluence URL is required.")
        return []

    if not username and not os.getenv("CONFLUENCE_USERNAME"):
        logger.error("Confluence username is required (or set CONFLUENCE_USERNAME env var).")
        return []
    
    if not api_token and not os.getenv("CONFLUENCE_API_TOKEN"):
        logger.error("Confluence API token is required (or set CONFLUENCE_API_TOKEN env var).")
        return []

    # Prefer explicitly passed credentials, then environment variables
    _username = username or os.getenv("CONFLUENCE_USERNAME")
    _api_token = api_token or os.getenv("CONFLUENCE_API_TOKEN")

    try:
        confluence = Confluence(
            url=confluence_url,
            username=_username,
            password=_api_token, # The python-atlassian library uses 'password' for token
            cloud=True # Assume cloud by default, user can set to False if on-prem with password
        )
    except Exception as e:
        logger.error(f"Failed to connect to Confluence at {confluence_url}: {e}")
        return []

    documents: List[str] = []
    page_ids_processed = set()

    try:
        if cql:
            logger.info(f"Fetching pages from Confluence using CQL: {cql} (limit: {limit})")
            results = confluence.cql(cql, limit=limit, expand="body.storage,version")
            pages = results.get('results', [])
        elif space_key:
            logger.info(f"Fetching pages from Confluence space: {space_key} (limit: {limit})")
            pages = confluence.get_all_pages_from_space(space_key, limit=limit, expand="body.storage,version")
        else:
            logger.error("Either 'space_key' or 'cql' must be provided.")
            return []

        if not pages:
            logger.warning("No pages found with the given criteria.")
            return []

        for page in pages:
            page_id = page['id']
            if page_id in page_ids_processed:
                continue
            page_ids_processed.add(page_id)

            page_title = page['title']
            logger.debug(f"Processing page: {page_title} (ID: {page_id})")
            
            # Extract page content (body is in HTML format)
            # We need to parse HTML to text. For simplicity, using regex to strip tags.
            # A more robust solution would use BeautifulSoup.
            try:
                page_content_html = page.get('body', {}).get('storage', {}).get('value', '')
                if page_content_html:
                    import re
                    text_content = re.sub('<[^<]+?>', '', page_content_html) # Basic HTML stripping
                    text_content = re.sub('\s+', ' ', text_content).strip() # Normalize whitespace
                    if text_content:
                        documents.append(f"Page Title: {page_title}\n\n{text_content}")
                else:
                    logger.warning(f"Page '{page_title}' has no body content.")
            except Exception as e:
                logger.error(f"Error extracting content from page '{page_title}': {e}")


            if include_attachments:
                logger.debug(f"Fetching attachments for page: {page_title} (ID: {page_id})")
                try:
                    attachments_container = confluence.get_attachments_from_content(
                        page_id=page_id, 
                        limit=attachment_limit_per_page,
                        expand="version" # Necessary for download link construction
                    )
                    attachments = attachments_container.get('results', [])
                    
                    for attachment in attachments:
                        attachment_title = attachment['title']
                        media_type = attachment.get('metadata', {}).get('mediaType', '')
                        download_link_suffix = attachment.get('_links', {}).get('download')
                        
                        if not download_link_suffix:
                            logger.warning(f"Attachment '{attachment_title}' on page '{page_title}' has no download link.")
                            continue

                        attachment_download_url = f"{confluence_url.rstrip('/')}{download_link_suffix}"
                        logger.debug(f"Processing attachment: {attachment_title} ({media_type})")

                        try:
                            # This requires authentication for the GET request as well
                            # The Confluence object handles this internally if using session-based auth
                            # or if the token allows direct downloads.
                            # For simplicity, we assume the 'confluence' object's session can fetch it.
                            # More robust: use requests with explicit auth if needed.
                            
                            # A simple way to get attachment content (may need adjustment based on auth method)
                            # response = confluence.request(path=download_link_suffix, method='GET', absolute=True)
                            # attachment_content_bytes = response.content

                            # The library's direct download methods are more reliable:
                            attachment_content_bytes = confluence.get_attachment_by_id(
                                content_id=page_id, # This seems to be content_id not attachment_id for this method
                                attachment_filename=attachment_title, # This is actually attachment_id or filename
                                download=True # This makes it return bytes
                            )

                            attachment_text = ""
                            if media_type == 'text/plain' or attachment_title.lower().endswith(".txt"):
                                attachment_text = attachment_content_bytes.decode('utf-8', errors='ignore')
                            elif media_type == 'application/pdf' or attachment_title.lower().endswith(".pdf"):
                                import io
                                import PyPDF2
                                with io.BytesIO(attachment_content_bytes) as pdf_file:
                                    reader = PyPDF2.PdfReader(pdf_file)
                                    for page_num in range(len(reader.pages)):
                                        attachment_text += reader.pages[page_num].extract_text() or ""
                            else:
                                logger.info(f"Skipping unsupported attachment type: {attachment_title} ({media_type})")
                                continue
                            
                            if attachment_text.strip():
                                documents.append(f"Attachment from Page '{page_title}': {attachment_title}\n\n{attachment_text.strip()}")
                            else:
                                logger.warning(f"Empty content extracted from attachment: {attachment_title}")
                        except Exception as e:
                            logger.error(f"Error processing attachment '{attachment_title}' on page '{page_title}': {e}")
                except Exception as e:
                    logger.error(f"Error fetching attachments for page '{page_title}': {e}")

    except Exception as e:
        logger.error(f"An error occurred while fetching documents from Confluence: {e}")

    if not documents:
        logger.warning("No documents or attachment contents extracted from Confluence.")
    else:
        logger.info(f"Successfully extracted {len(documents)} content items from Confluence.")
        
    return documents

# Example Usage (optional, for testing purposes)
# if __name__ == '__main__':
#     # Configure your Confluence details here or use environment variables
#     # CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN
#     # confluence_docs = get_documents_from_confluence(
#     #     confluence_url=os.getenv("CONFLUENCE_URL"), 
#     #     # username=os.getenv("CONFLUENCE_USERNAME"), # Loaded from env if None
#     #     # api_token=os.getenv("CONFLUENCE_API_TOKEN"), # Loaded from env if None
#     #     space_key="YOUR_SPACE_KEY", # Replace with a real space key
#     #     limit=5,
#     #     include_attachments=True
#     # )
#     # for i, doc_content in enumerate(confluence_docs):
#     #     print(f"--- Document {i+1} ---")
#     #     print(doc_content[:500] + "..." if len(doc_content) > 500 else doc_content)
#     pass
