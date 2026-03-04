"""
资源下载工具 - 对应 Java ResourceDownloadTool.java

使用 requests 库下载文件，代替 Java 的 Hutool HttpUtil。
"""
import os
import requests
import config
import logging

logger = logging.getLogger(__name__)


class ResourceDownloadTool:
    """资源下载工具 - 对应 Java ResourceDownloadTool"""

    def __init__(self):
        self.file_dir = os.path.join(config.FILE_SAVE_DIR, "download")

    def download_resource(self, url: str, fileName: str) -> str:
        """下载资源 - 对应 Java @Tool downloadResource"""
        file_path = os.path.join(self.file_dir, fileName)
        try:
            os.makedirs(self.file_dir, exist_ok=True)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=60, stream=True)
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return f"Resource downloaded successfully to: {file_path}"
        except Exception as e:
            return f"Error downloading resource: {e}"

    def download_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "downloadResource",
                "description": "Download a resource from a given URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL of the resource to download"},
                        "fileName": {"type": "string", "description": "Name of the file to save the downloaded resource"}
                    },
                    "required": ["url", "fileName"]
                }
            }
        }
