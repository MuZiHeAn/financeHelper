"""
PDF 生成工具 - 对应 Java PDFGenerationTool.java

使用 reportlab 代替 Java 的 iText。
reportlab 原生支持中文字体。
"""
import os
import config
import logging

logger = logging.getLogger(__name__)


class PDFGenerationTool:
    """PDF 生成工具 - 对应 Java PDFGenerationTool"""

    def __init__(self):
        self.file_dir = os.path.join(config.FILE_SAVE_DIR, "pdf")

    def generate_pdf(self, fileName: str, content: str) -> str:
        """生成 PDF 文件 - 对应 Java @Tool generatePDF"""
        file_path = os.path.join(self.file_dir, fileName)
        try:
            os.makedirs(self.file_dir, exist_ok=True)

            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont

            # 尝试注册中文字体
            try:
                # Windows 系统字体
                font_path = "C:/Windows/Fonts/simhei.ttf"
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont("SimHei", font_path))
                    font_name = "SimHei"
                else:
                    font_name = "Helvetica"  # 默认字体（不支持中文）
            except Exception:
                font_name = "Helvetica"

            c = canvas.Canvas(file_path, pagesize=A4)
            c.setFont(font_name, 12)

            # 将内容按行写入 PDF
            width, height = A4
            y = height - 50
            for line in content.split("\n"):
                if y < 50:  # 换页
                    c.showPage()
                    c.setFont(font_name, 12)
                    y = height - 50
                c.drawString(50, y, line)
                y -= 18

            c.save()
            return f"PDF generated successfully to: {file_path}"

        except Exception as e:
            return f"Error generating PDF: {e}"

    def generate_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "generatePDF",
                "description": "Generate a PDF file with given content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fileName": {"type": "string", "description": "Name of the file to save the generated PDF"},
                        "content": {"type": "string", "description": "Content to be included in the PDF"}
                    },
                    "required": ["fileName", "content"]
                }
            }
        }
