from langchain_community.agent_toolkits import FileManagementToolkit
from mcp.server.fastmcp import FastMCP
from tempfile import TemporaryDirectory
from typing import Any
import os
import sys
import logging
import uvicorn
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_work_directory() -> str:
    """
    Determines the working directory for file operations.
    
    Priority:
    1. Command line argument --work-dir
    2. Environment variable FILE_MANAGEMENT_WORK_DIR
    3. Default: Current workspace directory
    
    Returns:
        str: Path to the working directory
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='File Management MCP Server')
    parser.add_argument('--work-dir', type=str, help='Working directory for file operations')
    parser.add_argument('--port', type=int, default=8006, help='Port to run the server on (default: 8006)')
    args, _ = parser.parse_known_args()
    
    # Check command line argument first
    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
        logger.info(f"Using work directory from command line: {work_dir}")
        return work_dir, args.port
    
    # Check environment variable
    env_work_dir = os.getenv('FILE_MANAGEMENT_WORK_DIR')
    if env_work_dir:
        work_dir = os.path.abspath(env_work_dir)
        logger.info(f"Using work directory from environment variable: {work_dir}")
        return work_dir, args.port
    
    # Default to current workspace directory instead of temporary directory
    work_dir = os.getcwd()  # 현재 작업 디렉토리 사용
    logger.info(f"Using current workspace directory: {work_dir}")
    return work_dir, args.port

# Get the working directory
WORK_DIR, SERVER_PORT = get_work_directory()

# Ensure the directory exists
if not os.path.exists(WORK_DIR):
    try:
        os.makedirs(WORK_DIR, exist_ok=True)
        logger.info(f"Created working directory: {WORK_DIR}")
    except Exception as e:
        logger.error(f"Failed to create working directory {WORK_DIR}: {e}")
        sys.exit(1)

logger.info(f"File operations will be performed in: {WORK_DIR}")
logger.info(f"Server will run on port: {SERVER_PORT}")

def get_file_toolkit() -> FileManagementToolkit:
    """
    Creates and returns a FileManagementToolkit instance.
    
    Returns:
        FileManagementToolkit: Toolkit with file management tools
    """
    try:
        toolkit = FileManagementToolkit(root_dir=WORK_DIR)
        logger.info("FileManagementToolkit created successfully")
        return toolkit
    except Exception as e:
        logger.error(f"Failed to create FileManagementToolkit: {e}")
        raise

# Initialize FastMCP server
mcp = FastMCP(
    "FileManager",
    instructions="A file management tool that can read, write, copy, move, delete files and list directories."
)

logger.info("FastMCP server initialized")

@mcp.tool()
async def read_file(file_path: str) -> str:
    """
    Reads the contents of a file.
    
    Args:
        file_path (str): Path to the file to read
        
    Returns:
        str: Contents of the file
    """
    try:
        logger.info(f"Reading file: {file_path}")
        toolkit = get_file_toolkit()
        tools = toolkit.get_tools()
        read_tool = next(tool for tool in tools if tool.name == "read_file")
        
        result = read_tool.invoke({"file_path": file_path})
        logger.info(f"Successfully read file: {file_path}")
        return result
        
    except Exception as e:
        error_msg = f"Error reading file {file_path}: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"

@mcp.tool()
async def write_file(file_path: str, text: str) -> str:
    """
    Writes text to a file.
    
    Args:
        file_path (str): Path to the file to write
        text (str): Text content to write to the file
        
    Returns:
        str: Success message or error
    """
    try:
        logger.info(f"Writing to file: {file_path}")
        toolkit = get_file_toolkit()
        tools = toolkit.get_tools()
        write_tool = next(tool for tool in tools if tool.name == "write_file")
        
        result = write_tool.invoke({"file_path": file_path, "text": text})
        logger.info(f"Successfully wrote to file: {file_path}")
        return result
        
    except Exception as e:
        error_msg = f"Error writing to file {file_path}: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"

@mcp.tool()
async def list_directory(directory_path: str = "") -> str:
    """
    Lists files and directories in the specified directory.
    
    Args:
        directory_path (str): Path to the directory to list (empty for current working directory)
        
    Returns:
        str: List of files and directories
    """
    try:
        logger.info(f"Listing directory: {directory_path if directory_path else 'current directory'}")
        toolkit = get_file_toolkit()
        tools = toolkit.get_tools()
        list_tool = next(tool for tool in tools if tool.name == "list_directory")
        
        result = list_tool.invoke({"directory_path": directory_path})
        logger.info("Successfully listed directory contents")
        return result
        
    except Exception as e:
        error_msg = f"Error listing directory {directory_path}: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"

@mcp.tool()
async def copy_file(source_path: str, destination_path: str) -> str:
    """
    Copies a file from source to destination.
    
    Args:
        source_path (str): Path to the source file
        destination_path (str): Path to the destination file
        
    Returns:
        str: Success message or error
    """
    try:
        logger.info(f"Copying file from {source_path} to {destination_path}")
        toolkit = get_file_toolkit()
        tools = toolkit.get_tools()
        copy_tool = next(tool for tool in tools if tool.name == "copy_file")
        
        result = copy_tool.invoke({"source_path": source_path, "destination_path": destination_path})
        logger.info(f"Successfully copied file from {source_path} to {destination_path}")
        return result
        
    except Exception as e:
        error_msg = f"Error copying file from {source_path} to {destination_path}: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"

@mcp.tool()
async def move_file(source_path: str, destination_path: str) -> str:
    """
    Moves a file from source to destination.
    
    Args:
        source_path (str): Path to the source file
        destination_path (str): Path to the destination file
        
    Returns:
        str: Success message or error
    """
    try:
        logger.info(f"Moving file from {source_path} to {destination_path}")
        toolkit = get_file_toolkit()
        tools = toolkit.get_tools()
        move_tool = next(tool for tool in tools if tool.name == "move_file")
        
        result = move_tool.invoke({"source_path": source_path, "destination_path": destination_path})
        logger.info(f"Successfully moved file from {source_path} to {destination_path}")
        return result
        
    except Exception as e:
        error_msg = f"Error moving file from {source_path} to {destination_path}: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"

@mcp.tool()
async def delete_file(file_path: str) -> str:
    """
    Deletes a file.
    
    Args:
        file_path (str): Path to the file to delete
        
    Returns:
        str: Success message or error
    """
    try:
        logger.info(f"Deleting file: {file_path}")
        toolkit = get_file_toolkit()
        tools = toolkit.get_tools()
        delete_tool = next(tool for tool in tools if tool.name == "delete_file")
        
        result = delete_tool.invoke({"file_path": file_path})
        logger.info(f"Successfully deleted file: {file_path}")
        return result
        
    except Exception as e:
        error_msg = f"Error deleting file {file_path}: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"

@mcp.tool()
async def search_files(pattern: str, directory_path: str = "") -> str:
    """
    Searches for files matching a pattern.
    
    Args:
        pattern (str): Search pattern (filename or regex)
        directory_path (str): Directory to search in (empty for current working directory)
        
    Returns:
        str: List of matching files
    """
    try:
        logger.info(f"Searching for files with pattern: {pattern}")
        toolkit = get_file_toolkit()
        tools = toolkit.get_tools()
        search_tool = next(tool for tool in tools if tool.name == "file_search")
        
        search_params = {"pattern": pattern}
        if directory_path:
            search_params["directory_path"] = directory_path
            
        result = search_tool.invoke(search_params)
        logger.info(f"Successfully searched for files with pattern: {pattern}")
        return result
        
    except Exception as e:
        error_msg = f"Error searching for files with pattern {pattern}: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"


if __name__ == "__main__":
    logger.info("Starting file_management MCP server...")
    logger.info(f"Working directory: {WORK_DIR}")
    logger.info(f"Server port: {SERVER_PORT}")
    
    try:
        # Get the SSE app and run it on the specified port
        app = mcp.sse_app()
        uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)