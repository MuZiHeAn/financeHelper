"""
文件操作工具 - 对应 Java FileOperationTool.java

提供文件的读取、写入、追加三个功能。
使用 Python pathlib 实现文件操作。
"""
import os
from pathlib import Path
import config


class FileOperationTool:
    """文件操作工具 - 对应 Java FileOperationTool"""

    def __init__(self):
        self.file_dir = os.path.join(config.FILE_SAVE_DIR, "file")

    def read_file(self, fileName: str) -> str:
        """读取文件内容 - 对应 Java @Tool readFile"""
        file_path = Path(self.file_dir) / fileName
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"

    def write_file(self, fileName: str, content: str) -> str:
        """写入文件内容 - 对应 Java @Tool writeFile"""
        file_path = Path(self.file_dir) / fileName
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return f"File written successfully to: {file_path}"
        except Exception as e:
            return f"Error writing to file: {e}"

    def append_to_file(self, fileName: str, content: str) -> str:
        """追加内容到文件 - 对应 Java @Tool appendToFile"""
        file_path = Path(self.file_dir) / fileName
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(content)
            return f"Content appended successfully to: {file_path}"
        except Exception as e:
            return f"Error appending to file: {e}"

    # ===== 工具 JSON Schema 定义（给大模型看的）=====

    def read_file_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "readFile",
                "description": "Read content from a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fileName": {"type": "string", "description": "Name of the file to read"}
                    },
                    "required": ["fileName"]
                }
            }
        }

    def write_file_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "writeFile",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fileName": {"type": "string", "description": "Name of the file to write"},
                        "content": {"type": "string", "description": "Content to write to the file"}
                    },
                    "required": ["fileName", "content"]
                }
            }
        }

    def append_to_file_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "appendToFile",
                "description": "Append content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fileName": {"type": "string", "description": "Name of the file to append to"},
                        "content": {"type": "string", "description": "Content to append to the file"}
                    },
                    "required": ["fileName", "content"]
                }
            }
        }
