#!/usr/bin/env python3
"""Render REPORT.md as a paginated, selectable-text PDF using Markdown and ReportLab.

Math delimiters are removed and the expressions are emitted as readable source
text; this renderer does not typeset LaTeX. Markdown tables, headings, lists,
code blocks, links, and local raster images are handled explicitly.
"""

from __future__ import annotations

import argparse
import html
import re
from html.parser import HTMLParser
from pathlib import Path

import markdown2
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent.parent


def plain_math(text: str) -> str:
    """Keep equations legible as text when converting LaTeX-like Markdown."""
    text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\(?:mathbf|text|mathrm|mathcal|bar)\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\frac\{(.*?)\}\{(.*?)\}", r"(\1)/(\2)", text)
    text = text.replace(r"\nabla", "∇").replace(r"\theta", "θ").replace(r"\mathcal", "")
    text = re.sub(r"\\(?:pi|Delta|odot|in|to|times|cdot|sum|sigma|tau|mathbf)", "", text)
    return text.replace("$$", "").replace("$", "")


def inline_markup(fragment: str) -> str:
    fragment = plain_math(fragment)
    fragment = re.sub(r"<img\b[^>]*>", "[image]", fragment, flags=re.I)
    fragment = re.sub(r"<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", r'<link href="\1" color="#1558a6">\2</link>', fragment, flags=re.I | re.S)
    fragment = re.sub(r"<del>(.*?)</del>", r"<strike>\1</strike>", fragment, flags=re.I | re.S)
    return fragment


class MarkdownFlowParser(HTMLParser):
    """Small block parser for markdown2's predictable HTML output."""

    def __init__(self, styles, base_dir: Path, available_width: float):
        super().__init__(convert_charrefs=True)
        self.styles = styles
        self.base_dir = base_dir
        self.available_width = available_width
        self.flowables = []
        self.stack = []
        self.text = []
        self.table_rows = []
        self.table_row = None
        self.list_items = []
        self.in_pre = False
        self.pre_text = []
        self.in_cell = False
        self.cell_text = []

    def _flush_para(self):
        value = "".join(self.text).strip()
        self.text.clear()
        if value:
            self.flowables.append(Paragraph(inline_markup(value), self.styles["BodyTextCustom"]))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush_para()
            self.stack.append(("heading", tag))
        elif tag == "p":
            self._flush_para()
            self.stack.append(("p", tag))
        elif tag == "pre":
            self._flush_para()
            self.in_pre = True
            self.pre_text = []
        elif tag == "table":
            self._flush_para()
            self.table_rows = []
        elif tag == "tr":
            self.table_row = []
        elif tag in ("th", "td"):
            self.in_cell = True
            self.cell_text = []
        elif tag in ("ul", "ol"):
            self._flush_para()
            self.list_items = []
            self.stack.append(("list", tag, attrs.get("start", "1")))
        elif tag == "li":
            self._flush_para()
            self.stack.append(("li", tag))
        elif tag == "blockquote":
            self._flush_para()
            self.stack.append(("blockquote", tag))
        elif tag == "img":
            self._flush_para()
            src = attrs.get("src", "")
            if src and not src.startswith(("http://", "https://", "data:")):
                image_path = (self.base_dir / src).resolve()
                if image_path.exists():
                    image = Image(str(image_path))
                    scale = min(1.0, self.available_width / image.drawWidth, 4.2 * inch / image.drawHeight)
                    image.drawWidth *= scale
                    image.drawHeight *= scale
                    self.flowables.extend([Spacer(1, 5), image, Spacer(1, 8)])
        elif tag in ("strong", "b", "em", "i", "code", "s", "del"):
            mapped = {"strong": "b", "em": "i", "s": "strike", "del": "strike"}.get(tag, tag)
            target = self.cell_text if self.in_cell else self.text
            target.append(f"<{mapped}>")
        elif tag == "a":
            target = self.cell_text if self.in_cell else self.text
            href = html.escape(attrs.get("href", ""), quote=True)
            target.append(f'<link href="{href}" color="#1558a6">')
        elif tag == "br":
            (self.cell_text if self.in_cell else self.text).append("<br/>")
        elif tag == "hr":
            self._flush_para()
            self.flowables.extend([Spacer(1, 5), Table([['']], colWidths=[self.available_width], rowHeights=[1], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#b8c2cc"))])), Spacer(1, 8)])

    def handle_endtag(self, tag):
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            value = "".join(self.text).strip()
            self.text.clear()
            level = int(tag[1])
            if value:
                self.flowables.append(Paragraph(inline_markup(value), self.styles[f"Heading{min(level, 3)}Custom"]))
        elif tag == "p":
            self._flush_para()
        elif tag == "pre":
            value = plain_math("".join(self.pre_text)).strip("\n")
            self.pre_text = []
            self.in_pre = False
            if value:
                self.flowables.append(Preformatted(value, self.styles["CodeCustom"], maxLineLength=100))
                self.flowables.append(Spacer(1, 5))
        elif tag in ("th", "td") and self.in_cell:
            self.table_row.append(Paragraph(inline_markup("".join(self.cell_text).strip()), self.styles["TableHeader" if tag == "th" else "TableCell"]))
            self.cell_text = []
            self.in_cell = False
        elif tag == "tr" and self.table_row is not None:
            self.table_rows.append(self.table_row)
            self.table_row = None
        elif tag == "table":
            if self.table_rows:
                cols = max(len(row) for row in self.table_rows)
                widths = [self.available_width / cols] * cols
                table = Table(self.table_rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#15283d")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aeb8c2")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                self.flowables.extend([table, Spacer(1, 8)])
            self.table_rows = []
        elif tag == "li":
            value = "".join(self.text).strip()
            self.text.clear()
            if value:
                self.list_items.append(ListItem(Paragraph(inline_markup(value), self.styles["BodyTextCustom"]), leftIndent=10))
        elif tag in ("ul", "ol"):
            if self.list_items:
                ordered = tag == "ol"
                self.flowables.append(ListFlowable(self.list_items, bulletType="1" if ordered else "bullet", start="1", leftIndent=20, bulletFontName="DejaVu"))
                self.flowables.append(Spacer(1, 4))
            self.list_items = []
        elif tag == "blockquote":
            value = "".join(self.text).strip()
            self.text.clear()
            if value:
                self.flowables.append(Paragraph(inline_markup(value), self.styles["QuoteCustom"]))
        elif tag in ("strong", "b", "em", "i", "code", "s", "del", "a"):
            mapped = {"strong": "b", "em": "i", "s": "strike", "del": "strike", "a": "link"}.get(tag, tag)
            (self.cell_text if self.in_cell else self.text).append(f"</{mapped}>")

    def handle_data(self, data):
        if self.in_pre:
            self.pre_text.append(data)
        elif self.in_cell:
            self.cell_text.append(html.escape(data))
        else:
            self.text.append(html.escape(data))


def build_styles():
    font_path = Path(__import__("matplotlib").get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    bold_path = font_path.with_name("DejaVuSans-Bold.ttf")
    pdfmetrics.registerFont(TTFont("DejaVu", str(font_path)))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(bold_path)))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu", boldItalic="DejaVu-Bold")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyTextCustom", parent=styles["BodyText"], fontName="DejaVu", fontSize=8.5, leading=12, spaceAfter=6, textColor=colors.HexColor("#202b36")))
    styles.add(ParagraphStyle(name="Heading1Custom", parent=styles["Heading1"], fontName="DejaVu-Bold", fontSize=18, leading=22, spaceBefore=14, spaceAfter=9, keepWithNext=True, textColor=colors.HexColor("#143654")))
    styles.add(ParagraphStyle(name="Heading2Custom", parent=styles["Heading2"], fontName="DejaVu-Bold", fontSize=13, leading=17, spaceBefore=11, spaceAfter=6, keepWithNext=True, textColor=colors.HexColor("#1f5076")))
    styles.add(ParagraphStyle(name="Heading3Custom", parent=styles["Heading3"], fontName="DejaVu-Bold", fontSize=10, leading=13, spaceBefore=8, spaceAfter=4, keepWithNext=True, textColor=colors.HexColor("#2f5d7c")))
    styles.add(ParagraphStyle(name="TableCell", fontName="DejaVu", fontSize=6.5, leading=8, spaceAfter=0, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="TableHeader", fontName="DejaVu-Bold", fontSize=6.5, leading=8, spaceAfter=0, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="CodeCustom", fontName="DejaVu", fontSize=6.5, leading=8, leftIndent=5, rightIndent=5, backColor=colors.HexColor("#f3f5f7"), borderPadding=5))
    styles.add(ParagraphStyle(name="QuoteCustom", parent=styles["BodyTextCustom"], leftIndent=12, borderColor=colors.HexColor("#527b9d"), borderWidth=1, borderPadding=5, backColor=colors.HexColor("#f5f8fa")))
    return styles


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 7)
    canvas.setFillColor(colors.HexColor("#677788"))
    canvas.drawString(doc.leftMargin, 0.42 * inch, "Temporal Visual Intelligence | Research report draft")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "REPORT.md")
    parser.add_argument("--output", type=Path, default=ROOT / "report.pdf")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    text = source.read_text(encoding="utf-8")
    rendered = markdown2.markdown(text, extras=["tables", "fenced-code-blocks", "strike", "task_list"])
    styles = build_styles()
    doc = SimpleDocTemplate(str(output), pagesize=letter, rightMargin=0.65*inch, leftMargin=0.65*inch, topMargin=0.62*inch, bottomMargin=0.68*inch, title="Temporal Visual Intelligence Research Report", author="Temporal Visual Intelligence")
    flow = MarkdownFlowParser(styles, source.parent, doc.width)
    flow.feed(rendered)
    flow.close()
    doc.build(flow.flowables, onFirstPage=page_footer, onLaterPages=page_footer)
    print(f"Rendered {source.relative_to(ROOT) if source.is_relative_to(ROOT) else source} -> {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
