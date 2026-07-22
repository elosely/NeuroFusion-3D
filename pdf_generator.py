import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class ClinicalPDFReport:
    """
    Automated Medical PDF Report Generator for NeuroFusion 3D CDSS.
    """

    @staticmethod
    def generate_report(meta: dict, vol_stats: dict) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        
        # Custom Paragraph Styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=20,
            textColor=colors.HexColor("#1A365D"),
            spaceAfter=12
        )
        
        subtitle_style = ParagraphStyle(
            'SubTitleStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=20
        )

        section_heading = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=colors.HexColor("#2B6CB0"),
            spaceBefore=12,
            spaceAfter=8
        )

        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=colors.HexColor("#2D3748"),
            leading=14
        )

        elements = []

        # 1. Header
        elements.append(Paragraph("NeuroFusion 3D — Clinical Diagnostic Report", title_style))
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"<b>Generated On:</b> {now_str} | <b>Software Version:</b> v2.1-Clinical", subtitle_style))
        elements.append(Spacer(1, 10))

        # 2. Imaging Parameters Table
        elements.append(Paragraph("1. Scan Technical Parameters", section_heading))
        
        spacing_str = f"{meta['voxel_dimensions_mm'][0]:.2f} x {meta['voxel_dimensions_mm'][1]:.2f} x {meta['voxel_dimensions_mm'][2]:.2f} mm"
        dim_str = f"{meta['shape'][0]} x {meta['shape'][1]} x {meta['shape'][2]}"

        tech_data = [
            ["Parameter", "Value"],
            ["File Format", meta.get("format", "NIfTI")],
            ["3D Matrix Shape", dim_str],
            ["Voxel Dimensions (Spacing)", spacing_str],
            ["Single Voxel Physical Volume", f"{meta.get('voxel_volume_cm3', 0):.6f} cm³"]
        ]

        t_tech = Table(tech_data, colWidths=[200, 300])
        t_tech.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F7FAFC")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
        ]))
        elements.append(t_tech)
        elements.append(Spacer(1, 15))

        # 3. Volumetric Quantification Table
        elements.append(Paragraph("2. Quantitative Tumor Mass Assessment", section_heading))

        vol_data = [
            ["Clinical Metric", "Quantitative Measurement"],
            ["Total Positive Tumor Voxels", f"{vol_stats['total_positive_voxels']:,}"],
            ["Calculated Anatomical Tumor Mass", f"{vol_stats['calculated_volume_cm3']} cm³"]
        ]

        t_vol = Table(vol_data, colWidths=[200, 300])
        t_vol.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#C53030")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#FFF5F5")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#FEB2B2")),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
        ]))
        elements.append(t_vol)
        elements.append(Spacer(1, 15))

        # 4. Clinical Impression
        elements.append(Paragraph("3. Clinical Impression & Summary", section_heading))
        impression_text = (
            f"Volumetric 3D MRI analysis indicates a localized tissue hyperintensity region. "
            f"The total segmented lesion volume is estimated at <b>{vol_stats['calculated_volume_cm3']} cm³</b>. "
            "This automated measurement is derived using 3D spatial voxel dimensions and zero-mean normalized MRI data. "
            "<i>Note: This computer-assisted quantitative report is designed to support clinical radiologist assessment.</i>"
        )
        elements.append(Paragraph(impression_text, body_style))

        # Build Document
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()