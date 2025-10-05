"""Markdown to PDF converter utility."""
import markdown
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MarkdownToPDF:
    """Converter for markdown to PDF reports."""

    def __init__(self):
        """Initialize converter."""
        self.css_template = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                max-width: 850px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #34495e;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 5px;
                margin-top: 30px;
            }
            h3 {
                color: #7f8c8d;
                margin-top: 20px;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }
            th {
                background-color: #3498db;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            code {
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }
            blockquote {
                border-left: 4px solid #3498db;
                padding-left: 15px;
                color: #555;
                font-style: italic;
                margin: 15px 0;
            }
            .citation {
                color: #3498db;
                font-size: 0.9em;
            }
            .sentiment-positive {
                color: #27ae60;
                font-weight: bold;
            }
            .sentiment-negative {
                color: #e74c3c;
                font-weight: bold;
            }
            .sentiment-neutral {
                color: #95a5a6;
            }
            .metric-box {
                background-color: #ecf0f1;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
        </style>
        """

    def convert(
        self, 
        markdown_text: str, 
        output_path: str,
        title: str = "Financial Analysis Report"
    ) -> bool:
        """
        Convert markdown to PDF.
        
        Args:
            markdown_text: Markdown content
            output_path: Path to save PDF
            title: Document title
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert markdown to HTML
            html_body = markdown.markdown(
                markdown_text,
                extensions=['tables', 'fenced_code', 'codehilite']
            )
            
            # Create full HTML document
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{title}</title>
                {self.css_template}
            </head>
            <body>
                {html_body}
            </body>
            </html>
            """
            
            # Save HTML version (always works)
            html_path = output_path.replace('.pdf', '.html')
            Path(html_path).write_text(html, encoding='utf-8')
            logger.info(f"Saved HTML report to: {html_path}")
            
            # Try to convert to PDF if possible
            try:
                import pdfkit
                pdfkit.from_string(html, output_path)
                logger.info(f"Saved PDF report to: {output_path}")
            except ImportError:
                logger.warning("pdfkit not available, PDF conversion skipped. Install wkhtmltopdf for PDF support.")
            except Exception as e:
                logger.warning(f"PDF conversion failed: {e}. HTML version saved instead.")
            
            return True
        except Exception as e:
            logger.error(f"Error converting markdown to PDF: {e}")
            return False

    def save_markdown(self, content: str, output_path: str) -> bool:
        """
        Save markdown content to file.
        
        Args:
            content: Markdown content
            output_path: Path to save file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            Path(output_path).write_text(content, encoding='utf-8')
            logger.info(f"Saved markdown to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving markdown: {e}")
            return False
