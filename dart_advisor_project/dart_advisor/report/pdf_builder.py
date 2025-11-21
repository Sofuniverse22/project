"""Build PDF reports using ReportLab"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from pathlib import Path
from datetime import datetime
import logging
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


class PDFBuilder:
    """Build professional PDF reports"""

    def __init__(self, output_path: Path):
        """
        Initialize PDF builder

        Args:
            output_path: Path for output PDF
        """
        self.output_path = Path(output_path)
        self.story = []

        # Create document
        self.doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=2.5*cm,
            rightMargin=2.5*cm,
            topMargin=2.5*cm,
            bottomMargin=2.5*cm
        )

        # Setup styles
        self.styles = self._setup_styles()

    def _setup_styles(self):
        """Setup paragraph styles"""
        styles = getSampleStyleSheet()

        # Helper to safely add style
        def safe_add_style(styles, name, **kwargs):
            if name not in styles.byName:
                styles.add(ParagraphStyle(name=name, **kwargs))

        # Title style
        safe_add_style(styles, 'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        # Subtitle style
        safe_add_style(styles, 'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#555555'),
            spaceAfter=20,
            alignment=TA_CENTER
        )

        # Section header
        safe_add_style(styles, 'SectionHeader',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        )

        # Subsection header
        safe_add_style(styles, 'SubsectionHeader',
            parent=styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#333333'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )

        # Body text
        safe_add_style(styles, 'BodyText',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=10
        )

        return styles

    def add_cover_page(self, company_name: str, report_title: str = "Investment Analysis Report"):
        """
        Add cover page

        Args:
            company_name: Company name
            report_title: Report title
        """
        # Title
        self.story.append(Spacer(1, 5*cm))
        self.story.append(Paragraph(report_title, self.styles['CustomTitle']))
        self.story.append(Spacer(1, 1*cm))

        # Company name
        self.story.append(Paragraph(company_name, self.styles['CustomSubtitle']))
        self.story.append(Spacer(1, 3*cm))

        # Date
        date_str = datetime.now().strftime("%Y년 %m월 %d일")
        self.story.append(Paragraph(date_str, self.styles['CustomSubtitle']))

        # Page break
        self.story.append(PageBreak())

    def add_section(self, title: str, content: str, level: int = 2):
        """
        Add a section with title and content

        Args:
            title: Section title
            content: Section content
            level: Header level (2 or 3)
        """
        # Add title
        style_name = 'SectionHeader' if level == 2 else 'SubsectionHeader'
        self.story.append(Paragraph(title, self.styles[style_name]))
        self.story.append(Spacer(1, 0.3*cm))

        # Skip if content is None or empty
        if not content:
            return

        # Add content (split into paragraphs)
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # Clean up the text
                para = para.strip().replace('\n', '<br/>')
                self.story.append(Paragraph(para, self.styles['BodyText']))
                self.story.append(Spacer(1, 0.2*cm))

    def add_chart(
        self,
        image_path: Path,
        width: float = 15*cm,
        caption: str = None
    ):
        """
        Add chart image

        Args:
            image_path: Path to image file
            width: Image width
            caption: Optional caption
        """
        if not Path(image_path).exists():
            logger.warning(f"Chart image not found: {image_path}")
            return

        # Get actual image dimensions
        with PILImage.open(image_path) as pil_img:
            img_width, img_height = pil_img.size

        # Calculate height maintaining aspect ratio
        aspect_ratio = img_height / img_width
        height = width * aspect_ratio

        # Create image with explicit width and height
        img = Image(str(image_path), width=width, height=height)

        # Add image
        self.story.append(img)

        # Add caption if provided
        if caption:
            caption_style = ParagraphStyle(
                'Caption',
                parent=self.styles['Normal'],
                fontSize=10,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#666666')
            )
            self.story.append(Spacer(1, 0.2*cm))
            self.story.append(Paragraph(caption, caption_style))

        self.story.append(Spacer(1, 0.5*cm))

    def add_table(
        self,
        data: list,
        col_widths: list = None,
        header_row: bool = True
    ):
        """
        Add table

        Args:
            data: Table data (list of lists)
            col_widths: Column widths
            header_row: Whether first row is header
        """
        if not data:
            return

        # Create table
        table = Table(data, colWidths=col_widths)

        # Style
        style_commands = [
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]

        if header_row:
            style_commands.extend([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ])

        table.setStyle(TableStyle(style_commands))

        self.story.append(table)
        self.story.append(Spacer(1, 0.5*cm))

    def add_bullet_list(self, items: list):
        """
        Add bullet list

        Args:
            items: List of bullet points
        """
        for item in items:
            bullet_para = Paragraph(f"• {item}", self.styles['BodyText'])
            self.story.append(bullet_para)

    def add_page_break(self):
        """Add page break"""
        self.story.append(PageBreak())

    def add_spacer(self, height: float = 1*cm):
        """
        Add vertical space

        Args:
            height: Height of spacer
        """
        self.story.append(Spacer(1, height))

    def build(self):
        """Build the PDF"""
        try:
            self.doc.build(self.story)
            logger.info(f"PDF built successfully: {self.output_path}")
        except Exception as e:
            logger.error(f"Error building PDF: {e}")
            raise
