"""
文档加载器 - 对应 Java LoveAppDocumentLoader.java

加载 documents/ 目录下的 Markdown 文件，解析为 LangChain Document 对象。
从文件名中提取 status 元数据（如"单身"、"恋爱"、"已婚"）。

【对应 Java 逻辑】
- @Value("classpath:documents/*.md") → Python glob 扫描
- MarkdownDocumentReader → LangChain TextLoader + 自定义分割
- withAdditionalMetadata("status", status) → Document.metadata
"""
import os
import glob
import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

logger = logging.getLogger(__name__)


def extract_status_from_filename(filename: str) -> str:
    """
    从文件名中提取 status 元数据 - 对应 Java 中的文件名解析逻辑

    文件名格式示例: "恋爱宝典 - 单身篇.md"
    提取规则: 取最后一部分去掉.md后，倒数第三和倒数第二个字
    """
    try:
        if " - " in filename:
            parts = filename.split(" - ")
            last_part = parts[-1]
            base_name = last_part.replace(".md", "")
            if len(base_name) >= 3:
                return base_name[-3:-1]
    except Exception:
        pass
    return "unknown"


def load_markdown_documents() -> list[Document]:
    """
    加载所有 Markdown 文件 - 对应 Java LoveAppDocumentLoader.loadMarkdown()

    流程:
    1. 扫描 documents/ 目录下所有 .md 文件
    2. 读取每个文件的内容
    3. 按照 "---" 分割符分割为多个文档片段
    4. 为每个片段添加元数据（filename, status）

    Returns:
        Document 对象列表
    """
    documents_dir = config.DOCUMENTS_DIR
    all_documents: list[Document] = []

    if not os.path.exists(documents_dir):
        logger.warning(f"文档目录不存在: {documents_dir}")
        return all_documents

    md_files = glob.glob(os.path.join(documents_dir, "*.md"))
    logger.info(f"找到 {len(md_files)} 个 Markdown 文件")

    for md_file in md_files:
        try:
            filename = os.path.basename(md_file)
            status = extract_status_from_filename(filename)

            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            # 按照水平分割线 "---" 分割文档（对应 Java 的 withHorizontalRuleCreateDocument(true)）
            sections = content.split("---")

            for section in sections:
                section = section.strip()
                if not section:
                    continue

                doc = Document(
                    page_content=section,
                    metadata={
                        "filename": filename,
                        "status": status,
                        "source": md_file,
                    }
                )
                all_documents.append(doc)

            logger.info(f"  已加载: {filename} (status={status}, {len(sections)} 个片段)")

        except Exception as e:
            logger.error(f"加载文件失败: {md_file}, error: {e}")

    logger.info(f"共加载 {len(all_documents)} 个文档片段")
    return all_documents
