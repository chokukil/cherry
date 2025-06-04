from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from typing import Any, List, Dict
import os
import sys
import logging
import uvicorn
import json
import hashlib
import glob
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set dummy API key before trying to load from .env
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-key-for-testing-connection-only")

# Load environment variables from .env file (contains API keys)
# This will only override if .env file exists and has the key
try:
    load_dotenv(override=True)
    logger.info("Successfully loaded .env file")
except Exception as e:
    logger.warning(f"Failed to load .env file: {e}")

# Check for OpenAI API key with fallback for testing
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your_") or OPENAI_API_KEY == "":
    logger.warning("OpenAI API key not found or using placeholder value.")
    logger.warning("Setting dummy key for testing - actual retrieval will fail without real key.")
    os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-testing-connection-only"
else:
    logger.info("Found OpenAI API key")

# Paths
PDF_DIR = "data/pdf"
VECTORSTORE_DIR = "data/vectorstores"
MANAGEMENT_FILE = os.path.join(PDF_DIR, "processing_status.json")
VECTORSTORE_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "combined_index")

class PDFManager:
    """PDF 파일 처리 상태를 관리하는 클래스"""
    
    def __init__(self):
        self.management_data = self.load_management_data()
    
    def load_management_data(self) -> Dict:
        """관리 파일을 로드하거나 새로 생성"""
        if os.path.exists(MANAGEMENT_FILE):
            try:
                with open(MANAGEMENT_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load management file: {e}")
        
        return {
            "last_updated": datetime.now().isoformat(),
            "processed_files": {},
            "vectorstore_info": {
                "total_documents": 0,
                "last_rebuild": None
            }
        }
    
    def save_management_data(self):
        """관리 데이터를 파일에 저장"""
        try:
            os.makedirs(os.path.dirname(MANAGEMENT_FILE), exist_ok=True)
            self.management_data["last_updated"] = datetime.now().isoformat()
            with open(MANAGEMENT_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.management_data, f, indent=2, ensure_ascii=False)
            logger.info("Management data saved successfully")
        except Exception as e:
            logger.error(f"Failed to save management data: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """파일의 MD5 해시를 계산"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def get_pdf_files(self) -> List[str]:
        """PDF 디렉토리에서 모든 PDF 파일 목록을 반환"""
        pdf_pattern = os.path.join(PDF_DIR, "*.pdf")
        return glob.glob(pdf_pattern)
    
    def get_unprocessed_files(self) -> List[str]:
        """처리되지 않은 PDF 파일 목록을 반환"""
        all_pdf_files = self.get_pdf_files()
        unprocessed = []
        
        for pdf_file in all_pdf_files:
            file_name = os.path.basename(pdf_file)
            current_hash = self.get_file_hash(pdf_file)
            
            if file_name not in self.management_data["processed_files"]:
                unprocessed.append(pdf_file)
            elif self.management_data["processed_files"][file_name]["hash"] != current_hash:
                unprocessed.append(pdf_file)
                logger.info(f"File {file_name} has been modified, needs reprocessing")
        
        return unprocessed
    
    def mark_file_processed(self, file_path: str, doc_count: int):
        """파일을 처리됨으로 표시"""
        file_name = os.path.basename(file_path)
        file_hash = self.get_file_hash(file_path)
        
        self.management_data["processed_files"][file_name] = {
            "hash": file_hash,
            "processed_date": datetime.now().isoformat(),
            "document_count": doc_count,
            "file_size": os.path.getsize(file_path)
        }


def process_pdf_file(file_path: str) -> List[Any]:
    """단일 PDF 파일을 처리하여 문서 청크를 반환"""
    try:
        logger.info(f"Processing PDF: {file_path}")
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        
        # 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=50
        )
        split_documents = text_splitter.split_documents(docs)
        
        # 메타데이터에 파일 정보 추가
        for doc in split_documents:
            doc.metadata['source_file'] = os.path.basename(file_path)
            doc.metadata['processed_date'] = datetime.now().isoformat()
        
        logger.info(f"Processed {file_path}: {len(docs)} pages -> {len(split_documents)} chunks")
        return split_documents
        
    except Exception as e:
        logger.error(f"Failed to process PDF {file_path}: {e}")
        raise


def create_or_update_vectorstore(new_documents: List[Any] = None) -> FAISS:
    """벡터스토어를 생성하거나 업데이트"""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 기존 벡터스토어가 있는지 확인
        if os.path.exists(os.path.join(VECTORSTORE_INDEX_PATH, "index.faiss")) and os.path.exists(os.path.join(VECTORSTORE_INDEX_PATH, "index.pkl")):
            logger.info("Loading existing vectorstore")
            vectorstore = FAISS.load_local(
                VECTORSTORE_INDEX_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # 새 문서가 있으면 추가
            if new_documents:
                logger.info(f"Adding {len(new_documents)} new documents to existing vectorstore")
                vectorstore.add_documents(new_documents)
                vectorstore.save_local(VECTORSTORE_INDEX_PATH)
        else:
            # 새 벡터스토어 생성
            if not new_documents:
                logger.warning("No documents to create vectorstore")
                return None
            
            logger.info(f"Creating new vectorstore with {len(new_documents)} documents")
            os.makedirs(VECTORSTORE_DIR, exist_ok=True)
            vectorstore = FAISS.from_documents(documents=new_documents, embedding=embeddings)
            vectorstore.save_local(VECTORSTORE_INDEX_PATH)
        
        logger.info("Vectorstore created/updated successfully")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Failed to create/update vectorstore: {e}")
        raise


def initialize_document_processing():
    """문서 처리 초기화 - 새로운 PDF 파일들을 자동으로 임베딩"""
    pdf_manager = PDFManager()
    
    # 처리되지 않은 파일들 확인
    unprocessed_files = pdf_manager.get_unprocessed_files()
    
    if not unprocessed_files:
        logger.info("All PDF files are already processed")
        return pdf_manager
    
    logger.info(f"Found {len(unprocessed_files)} unprocessed PDF files")
    
    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-dummy"):
        logger.warning("Cannot process new PDFs: Missing valid OpenAI API key")
        return pdf_manager
    
    all_new_documents = []
    
    # 각 파일 처리
    for pdf_file in unprocessed_files:
        try:
            documents = process_pdf_file(pdf_file)
            all_new_documents.extend(documents)
            pdf_manager.mark_file_processed(pdf_file, len(documents))
            logger.info(f"Successfully processed: {os.path.basename(pdf_file)}")
        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {e}")
            continue
    
    # 벡터스토어 업데이트
    if all_new_documents:
        try:
            create_or_update_vectorstore(all_new_documents)
            pdf_manager.management_data["vectorstore_info"]["last_rebuild"] = datetime.now().isoformat()
            pdf_manager.management_data["vectorstore_info"]["total_documents"] += len(all_new_documents)
            logger.info(f"Added {len(all_new_documents)} documents to vectorstore")
        except Exception as e:
            logger.error(f"Failed to update vectorstore: {e}")
    
    # 관리 데이터 저장
    pdf_manager.save_management_data()
    return pdf_manager


def create_retriever() -> Any:
    """저장된 벡터스토어로부터 리트리버를 생성"""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        if not os.path.exists(os.path.join(VECTORSTORE_INDEX_PATH, "index.faiss")):
            logger.error("No vectorstore found. Please process some PDF files first.")
            raise FileNotFoundError("No vectorstore found")
        
        logger.info("Loading vectorstore for retrieval")
        vectorstore = FAISS.load_local(
            VECTORSTORE_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # 상위 5개 문서 반환
        )
        
        logger.info("Retriever created successfully")
        return retriever
        
    except Exception as e:
        logger.error(f"Failed to create retriever: {e}")
        raise


# Initialize FastMCP server with configuration
mcp = FastMCP(
    "PrivateDocumentRetriever",
    instructions="A Retriever that can retrieve information from your personal document database built from multiple PDF files."
)

logger.info("FastMCP server initialized")

# 서버 시작시 문서 처리 초기화
pdf_manager = None

@mcp.tool()
async def retrieve(query: str) -> str:
    """
    Retrieves information from the personal document database based on the query.

    This function queries the pre-built vectorstore with the provided input,
    and returns the concatenated content of all retrieved documents.

    Args:
        query (str): The search query to find relevant information

    Returns:
        str: Concatenated text content from all retrieved documents with source information
    """
    try:
        logger.info(f"Received query: {query}")
        
        # Check for dummy API key and provide helpful error message
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-dummy"):
            return ("ERROR: Private RAG tool is running with dummy or missing OpenAI API key. "
                   "Please set a valid OPENAI_API_KEY in your .env file to use this tool. "
                   "However, the MCP server connection is working correctly.")
        
        # Create retriever from stored vectorstore
        logger.info("Creating retriever from stored vectorstore")
        retriever = create_retriever()

        # Use the invoke() method to get relevant documents based on the query
        logger.info("Invoking retriever with query")
        retrieved_docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        if not retrieved_docs:
            return "No relevant documents found for your query."

        # Format results with source information
        results = []
        for i, doc in enumerate(retrieved_docs):
            source_file = doc.metadata.get('source_file', 'Unknown')
            content = doc.page_content.strip()
            results.append(f"**Source {i+1}: {source_file}**\n{content}")
        
        result = "\n\n" + "="*50 + "\n\n".join(results)
        logger.info(f"Returning result with {len(retrieved_docs)} sources")
        return result
        
    except Exception as e:
        error_msg = f"Error in retrieve function: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"


@mcp.tool()
async def get_document_status() -> str:
    """
    Returns the current status of processed PDF documents and vectorstore information.

    Returns:
        str: JSON formatted status information
    """
    try:
        if pdf_manager:
            status = {
                "processed_files": len(pdf_manager.management_data["processed_files"]),
                "total_pdf_files": len(pdf_manager.get_pdf_files()),
                "vectorstore_info": pdf_manager.management_data["vectorstore_info"],
                "last_updated": pdf_manager.management_data["last_updated"],
                "file_details": pdf_manager.management_data["processed_files"]
            }
            return json.dumps(status, indent=2, ensure_ascii=False)
        else:
            return "PDF manager not initialized"
    except Exception as e:
        return f"Error getting document status: {str(e)}"


@mcp.tool()
async def refresh_documents() -> str:
    """
    Manually refresh and process any new or modified PDF files.

    Returns:
        str: Status message about the refresh operation
    """
    try:
        global pdf_manager
        logger.info("Manual document refresh requested")
        pdf_manager = initialize_document_processing()
        return "Document refresh completed successfully. Check logs for details."
    except Exception as e:
        error_msg = f"Error during document refresh: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"


if __name__ == "__main__":
    logger.info("Starting private_rag MCP server...")
    try:
        # Initialize document processing on startup
        logger.info("Initializing document processing...")
        pdf_manager = initialize_document_processing()
        logger.info("Document processing initialization completed")
        
        # Get the SSE app and run it on port 8005
        app = mcp.sse_app()
        uvicorn.run(app, host="0.0.0.0", port=8005)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)