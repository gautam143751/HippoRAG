from office365.runtime.auth.user_credential import UserCredential
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File
from typing import List, Optional, Dict
import io
import os

# For text extraction
import PyPDF2
import docx # python-docx
# Potentially: python-pptx for .pptx, openpyxl for .xlsx (if text extraction from them is desired)

# Configure logging
import logging
logger = logging.getLogger(__name__)

def get_documents_from_sharepoint(
    sharepoint_url: str, # e.g., "https://yourtenant.sharepoint.com/sites/YourSiteName"
    username: Optional[str] = None, # User email for SharePoint
    password: Optional[str] = None, # User password
    document_library_name: str = "Shared Documents", # Or other library name
    folder_path: Optional[str] = None, # Relative path within the library, e.g., "General/MyFolder"
    recursive: bool = True,
    limit: int = 100 # Limit on number of files to process
) -> List[str]:
    """
    Connects to SharePoint Online, fetches files from the specified document library and folder,
    extracts text content from supported file types (.txt, .pdf, .docx), and returns a list of strings.

    Args:
        sharepoint_url (str): The URL of the SharePoint site.
        username (Optional[str]): Username (email) for SharePoint authentication.
                                   If None, uses environment variable SHAREPOINT_USERNAME.
        password (Optional[str]): Password for SharePoint authentication.
                                   If None, uses environment variable SHAREPOINT_PASSWORD.
        document_library_name (str): The name or title of the document library.
        folder_path (Optional[str]): The relative path to the folder within the document library.
                                      If None, fetches from the root of the library.
        recursive (bool): Whether to fetch files from subfolders recursively.
        limit (int): Maximum number of files to attempt to process.

    Returns:
        List[str]: A list of strings, where each string is the text content of a document.
                   Returns an empty list on failure or if no supported files are found.
    """

    if not sharepoint_url:
        logger.error("SharePoint site URL is required.")
        return []

    _username = username or os.getenv("SHAREPOINT_USERNAME")
    _password = password or os.getenv("SHAREPOINT_PASSWORD")

    if not _username or not _password:
        logger.error("SharePoint username and password are required (or set SHAREPOINT_USERNAME/SHAREPOINT_PASSWORD env vars).")
        return []

    documents: List[str] = []
    processed_files_count = 0

    try:
        ctx = ClientContext(sharepoint_url).with_credentials(UserCredential(_username, _password))
        logger.info(f"Successfully authenticated with SharePoint site: {sharepoint_url}")

        # Construct the server-relative URL for the folder
        if folder_path:
            # Ensure folder_path doesn't start with a slash and document_library_name is clean
            clean_doc_lib = document_library_name.strip('/')
            clean_folder_path = folder_path.strip('/')
            # Server relative URL for a folder in a library is typically /sites/SiteName/LibraryName/FolderPath
            # The ClientContext site URL might already contain /sites/SiteName
            # We need to get the server relative URL of the library first
            web = ctx.web.get().execute_query()
            site_relative_url = web.server_relative_url 
            if not site_relative_url.endswith('/'):
                site_relative_url += '/'
            
            # Path to the folder within the library
            folder_server_relative_url = f"{site_relative_url.rstrip('/')}/{clean_doc_lib}/{clean_folder_path}"
        else:
            # Root of the document library
            web = ctx.web.get().execute_query()
            site_relative_url = web.server_relative_url
            if not site_relative_url.endswith('/'):
                site_relative_url += '/'
            folder_server_relative_url = f"{site_relative_url.rstrip('/')}/{document_library_name.strip('/')}"
        
        target_folder = ctx.web.get_folder_by_server_relative_url(folder_server_relative_url)
        ctx.load(target_folder)
        ctx.execute_query()
        logger.info(f"Accessing SharePoint folder: {target_folder.properties['ServerRelativeUrl']}")


        def process_folder(folder_obj, current_depth=0):
            nonlocal processed_files_count
            if processed_files_count >= limit:
                return

            # Expand files and folders for the current folder object
            ctx.load(folder_obj, ["Files", "Folders"])
            ctx.execute_query()
            
            # Process files in the current folder
            for file_item in folder_obj.files:
                if processed_files_count >= limit:
                    break
                
                file_name = file_item.properties["Name"]
                logger.debug(f"Processing file: {file_name} in folder {folder_obj.properties['ServerRelativeUrl']}")
                
                text_content = ""
                try:
                    # Download file content
                    file_content_bytes = File.open_binary(ctx, file_item.properties["ServerRelativeUrl"]).content
                    
                    if file_name.lower().endswith(".txt"):
                        text_content = file_content_bytes.decode('utf-8', errors='ignore')
                    elif file_name.lower().endswith(".pdf"):
                        with io.BytesIO(file_content_bytes) as pdf_file:
                            reader = PyPDF2.PdfReader(pdf_file)
                            for page_num in range(len(reader.pages)):
                                text_content += reader.pages[page_num].extract_text() or ""
                    elif file_name.lower().endswith(".docx"):
                        with io.BytesIO(file_content_bytes) as docx_file:
                            doc = docx.Document(docx_file)
                            for para in doc.paragraphs:
                                text_content += para.text + "\n"
                    else:
                        logger.info(f"Skipping unsupported file type: {file_name}")
                        continue
                    
                    if text_content.strip():
                        documents.append(f"SharePoint File: {file_name}\nPath: {file_item.properties['ServerRelativeUrl']}\n\n{text_content.strip()}")
                        processed_files_count += 1
                    else:
                        logger.warning(f"Empty content extracted from SharePoint file: {file_name}")

                except Exception as e:
                    logger.error(f"Error processing file {file_name} from SharePoint: {e}")
            
            # Process subfolders if recursive is True
            if recursive and current_depth < 10: # Max recursion depth to prevent infinite loops
                for sub_folder in folder_obj.folders:
                    if processed_files_count >= limit:
                        break
                    # Ensure sub_folder is loaded with its properties before accessing Name
                    ctx.load(sub_folder)
                    ctx.execute_query()
                    logger.debug(f"Entering subfolder: {sub_folder.properties['Name']}")
                    process_folder(sub_folder, current_depth + 1)

        process_folder(target_folder)

    except Exception as e:
        logger.error(f"An error occurred while fetching documents from SharePoint: {e}")
        # You might want to print traceback for debugging:
        # import traceback
        # logger.error(traceback.format_exc())


    if not documents:
        logger.warning(f"No documents extracted from SharePoint path: {sharepoint_url} / {document_library_name} / {folder_path or '(root)'}")
    else:
        logger.info(f"Successfully extracted {len(documents)} documents from SharePoint.")
            
    return documents

# Example Usage (optional, for testing purposes)
# if __name__ == '__main__':
#     # Configure your SharePoint details here or use environment variables
#     # SHAREPOINT_URL, SHAREPOINT_USERNAME, SHAREPOINT_PASSWORD
#     # sharepoint_docs = get_documents_from_sharepoint(
#     #     sharepoint_url=os.getenv("SHAREPOINT_URL"), # e.g. "https://yourtenant.sharepoint.com/sites/YourSite"
#     #     # username=os.getenv("SHAREPOINT_USERNAME"), # Loaded from env if None
#     #     # password=os.getenv("SHAREPOINT_PASSWORD"), # Loaded from env if None
#     #     document_library_name="Documents", # Or your specific library
#     #     folder_path="General", # Optional: path within the library
#     #     limit=10
#     # )
#     # for i, doc_content in enumerate(sharepoint_docs):
#     #     print(f"--- Document {i+1} ---")
#     #     print(doc_content[:500] + "..." if len(doc_content) > 500 else doc_content)
#     pass
