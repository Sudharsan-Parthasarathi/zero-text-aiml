"""
Pure Python PDF Generator for ZERO-TEXT Research Paper
Generates standard valid PDF 1.4 using built-in standard fonts (Helvetica, Helvetica-Bold, Helvetica-Oblique, Courier).
No external libraries or C-bindings required.
"""

import sys
import zlib
import re
from pathlib import Path

# A4 dimensions in points (72 points = 1 inch)
PAGE_WIDTH = 595.28
PAGE_HEIGHT = 841.89

# Margins
MARGIN_LEFT = 50.0
MARGIN_RIGHT = 50.0
MARGIN_TOP = 50.0
MARGIN_BOTTOM = 50.0
CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
CONTENT_HEIGHT = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

# Character width estimates for Helvetica (in units of 1/1000 of font size)
# Average width for Helvetica regular ~ 520, Bold ~ 560, Monospace Courier = 600
def get_char_width(char: str, font_name: str) -> float:
    if "Courier" in font_name:
        return 600.0
    if "Bold" in font_name:
        if char in "ijl|![]:;.,'":
            return 280.0
        elif char in "mwMW":
            return 850.0
        elif char in "ABCDEFGHJKLNOPQRSTUVXYZ":
            return 720.0
        elif char == " ":
            return 280.0
        else:
            return 580.0
    else:
        if char in "ijl|![]:;.,'":
            return 250.0
        elif char in "mwMW":
            return 820.0
        elif char in "ABCDEFGHJKLNOPQRSTUVXYZ":
            return 680.0
        elif char == " ":
            return 270.0
        else:
            return 530.0

def measure_text_width(text: str, font_name: str, font_size: float) -> float:
    total_units = sum(get_char_width(c, font_name) for c in text)
    return (total_units / 1000.0) * font_size

def escape_pdf_string(text: str) -> str:
    # Escape backslash, parentheses
    out = []
    for c in text:
        if c == '\\':
            out.append('\\\\')
        elif c == '(':
            out.append('\\(')
        elif c == ')':
            out.append('\\)')
        elif ord(c) < 128:
            out.append(c)
        else:
            # Map common unicode chars to ascii approximations
            m = {
                '–': '-', '—': '--', '’': "'", '‘': "'", '“': '"', '”': '"',
                '…': '...', '•': '*', '→': '->', '←': '<-', '↔': '<->',
                '≥': '>=', '≤': '<=', '≠': '!=', '≈': '~=', '×': 'x',
                '±': '+/-', '°': ' deg', 'µ': 'u', 'τ': 'tau', 'α': 'alpha',
                'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'σ': 'sigma',
                'λ': 'lambda', 'μ': 'mu', 'ρ': 'rho', 'θ': 'theta',
                '∑': 'sum', '∏': 'prod', '∈': ' in ', '∉': ' not in ',
                '⊂': ' subset ', '⊆': ' subseteq ', '∩': ' intersect ',
                '∪': ' union ', '∅': 'empty', '∞': 'inf', '∇': 'grad'
            }
            out.append(m.get(c, '?'))
    return "".join(out)

class PDFDocument:
    def __init__(self):
        self.objects = []
        self.pages = []
        
    def add_object(self, obj_bytes: bytes) -> int:
        self.objects.append(obj_bytes)
        return len(self.objects)

    def write_pdf(self, output_path: Path):
        # 1. Header
        header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        
        # 2. Page tree & font objects
        # Fonts: F1 = Helvetica, F2 = Helvetica-Bold, F3 = Helvetica-Oblique, F4 = Courier
        f1_idx = self.add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        f2_idx = self.add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        f3_idx = self.add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique /Encoding /WinAnsiEncoding >>")
        f4_idx = self.add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>")
        
        # Prepare Pages Object ID
        pages_obj_idx = len(self.objects) + len(self.pages) * 2 + 1
        
        page_refs = []
        # Add Page objects and Content stream objects
        for page_num, stream_content in enumerate(self.pages, 1):
            stream_compressed = zlib.compress(stream_content)
            stream_len = len(stream_compressed)
            stream_obj = (
                f"<< /Length {stream_len} /Filter /FlateDecode >>\nstream\n".encode("ascii")
                + stream_compressed
                + b"\nendstream"
            )
            content_idx = self.add_object(stream_obj)
            
            page_obj = (
                f"<< /Type /Page /Parent {pages_obj_idx} 0 R "
                f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Contents {content_idx} 0 R "
                f"/Resources << /Font << /F1 {f1_idx} 0 R /F2 {f2_idx} 0 R /F3 {f3_idx} 0 R /F4 {f4_idx} 0 R >> >> >>"
            ).encode("ascii")
            p_idx = self.add_object(page_obj)
            page_refs.append(f"{p_idx} 0 R")
            
        # Add Pages Object
        pages_content = (
            f"<< /Type /Pages /Kids [ {' '.join(page_refs)} ] /Count {len(page_refs)} >>"
        ).encode("ascii")
        actual_pages_idx = self.add_object(pages_content)
        
        # Add Catalog Object
        catalog_obj = f"<< /Type /Catalog /Pages {actual_pages_idx} 0 R >>".encode("ascii")
        catalog_idx = self.add_object(catalog_obj)
        
        # Write PDF to disk with Cross-Reference Table
        with open(output_path, "wb") as f:
            f.write(header)
            offsets = []
            curr_pos = len(header)
            
            for i, obj in enumerate(self.objects, 1):
                offsets.append(curr_pos)
                f.write(f"{i} 0 obj\n".encode("ascii"))
                f.write(obj)
                f.write(b"\nendobj\n")
                curr_pos = f.tell()
                
            xref_offset = curr_pos
            f.write(f"xref\n0 {len(self.objects) + 1}\n".encode("ascii"))
            f.write(b"0000000000 65535 f \n")
            for off in offsets:
                f.write(f"{off:010d} 00000 n \n".encode("ascii"))
                
            trailer = (
                f"trailer\n<< /Size {len(self.objects) + 1} /Root {catalog_idx} 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
            f.write(trailer)

class FlowableWriter:
    def __init__(self, doc: PDFDocument):
        self.doc = doc
        self.current_page_num = 1
        self.y_cursor = PAGE_HEIGHT - MARGIN_TOP
        self.commands = []
        
    def new_page(self):
        # Draw running header / footer on current page
        self.draw_page_decorations()
        # Commit stream
        stream = "\n".join(self.commands).encode("latin-1", errors="replace")
        self.doc.pages.append(stream)
        # Reset for next page
        self.current_page_num += 1
        self.y_cursor = PAGE_HEIGHT - MARGIN_TOP
        self.commands = []

    def draw_page_decorations(self):
        # Running top header rule and running bottom footer with page number
        if self.current_page_num > 1:
            header_cmd = (
                "0.7 0.7 0.7 RG 0.5 w\n"
                f"{MARGIN_LEFT} {PAGE_HEIGHT - 35} m {PAGE_WIDTH - MARGIN_RIGHT} {PAGE_HEIGHT - 35} l S\n"
                "BT /F3 8 Tf 0.3 0.3 0.3 rg "
                f"{MARGIN_LEFT} {PAGE_HEIGHT - 30} Td (ZERO-TEXT: Multimodal Retrieval & Sketch Projection Architecture) Tj ET"
            )
            self.commands.insert(0, header_cmd)
            
        footer_cmd = (
            "0.7 0.7 0.7 RG 0.5 w\n"
            f"{MARGIN_LEFT} 38 m {PAGE_WIDTH - MARGIN_RIGHT} 38 l S\n"
            f"BT /F1 9 Tf 0.2 0.2 0.2 rg "
            f"{PAGE_WIDTH / 2.0 - 15} 25 Td (Page {self.current_page_num}) Tj ET"
        )
        self.commands.append(footer_cmd)

    def ensure_space(self, height_needed: float):
        if (self.y_cursor - height_needed) < MARGIN_BOTTOM:
            self.new_page()

    def draw_heading(self, text: str, level: int):
        clean_text = escape_pdf_string(text)
        if level == 1:
            self.ensure_space(60)
            self.y_cursor -= 12
            # Center aligned large title
            font_size = 17.0
            font_name = "F2"
            w = measure_text_width(clean_text, "Helvetica-Bold", font_size)
            x_pos = (PAGE_WIDTH - w) / 2.0
            self.commands.append(f"BT /{font_name} {font_size} Tf 0.05 0.12 0.25 rg {x_pos:.2f} {self.y_cursor:.2f} Td ({clean_text}) Tj ET")
            self.y_cursor -= 10
            # Bottom title rule
            self.commands.append(f"0.05 0.12 0.25 RG 1.5 w {MARGIN_LEFT} {self.y_cursor:.2f} m {PAGE_WIDTH - MARGIN_RIGHT} {self.y_cursor:.2f} l S")
            self.y_cursor -= 18
        elif level == 2:
            self.ensure_space(42)
            self.y_cursor -= 14
            font_size = 12.5
            font_name = "F2"
            self.commands.append(f"BT /{font_name} {font_size} Tf 0.08 0.18 0.38 rg {MARGIN_LEFT:.2f} {self.y_cursor:.2f} Td ({clean_text}) Tj ET")
            self.y_cursor -= 4
            self.commands.append(f"0.7 0.75 0.85 RG 0.75 w {MARGIN_LEFT} {self.y_cursor:.2f} m {PAGE_WIDTH - MARGIN_RIGHT} {self.y_cursor:.2f} l S")
            self.y_cursor -= 12
        elif level == 3:
            self.ensure_space(32)
            self.y_cursor -= 10
            font_size = 10.5
            font_name = "F2"
            self.commands.append(f"BT /{font_name} {font_size} Tf 0.12 0.24 0.48 rg {MARGIN_LEFT:.2f} {self.y_cursor:.2f} Td ({clean_text}) Tj ET")
            self.y_cursor -= 10
        else:
            self.ensure_space(25)
            self.y_cursor -= 8
            font_size = 9.5
            font_name = "F2"
            self.commands.append(f"BT /{font_name} {font_size} Tf 0.15 0.28 0.52 rg {MARGIN_LEFT:.2f} {self.y_cursor:.2f} Td ({clean_text}) Tj ET")
            self.y_cursor -= 8

    def draw_paragraph(self, text: str, font_size: float = 9.5, line_height: float = 13.5, font_name: str = "F1", color: tuple = (0.1, 0.1, 0.1), indent: float = 0.0):
        # Word wrap text across available content width
        words = text.split()
        if not words:
            return
            
        cur_width = CONTENT_WIDTH - indent
        wrapped_lines = []
        cur_line = []
        cur_line_len = 0.0
        
        font_measure = "Helvetica-Bold" if "F2" in font_name else ("Helvetica-Oblique" if "F3" in font_name else "Helvetica")
        space_width = measure_text_width(" ", font_measure, font_size)
        
        for w in words:
            clean_w = escape_pdf_string(w)
            w_len = measure_text_width(clean_w, font_measure, font_size)
            if cur_line and (cur_line_len + space_width + w_len) > cur_width:
                wrapped_lines.append(" ".join(cur_line))
                cur_line = [clean_w]
                cur_line_len = w_len
            else:
                cur_line.append(clean_w)
                cur_line_len += (space_width if cur_line_len > 0 else 0.0) + w_len
                
        if cur_line:
            wrapped_lines.append(" ".join(cur_line))
            
        total_height = len(wrapped_lines) * line_height + 4
        self.ensure_space(total_height)
        
        r, g, b = color
        for line in wrapped_lines:
            self.y_cursor -= line_height
            x_pos = MARGIN_LEFT + indent
            self.commands.append(f"BT /{font_name} {font_size} Tf {r} {g} {b} rg {x_pos:.2f} {self.y_cursor:.2f} Td ({line}) Tj ET")
        self.y_cursor -= 4

    def draw_code_block(self, lines: list):
        font_size = 7.5
        line_height = 9.8
        pad_x = 8.0
        pad_y = 6.0
        total_lines = len(lines)
        total_height = total_lines * line_height + pad_y * 2
        
        self.ensure_space(min(total_height, 150))
        
        # Draw background container
        box_y = self.y_cursor - total_height
        self.commands.append(
            f"0.96 0.97 0.98 rg 0.8 0.84 0.88 RG 0.5 w "
            f"{MARGIN_LEFT} {box_y:.2f} {CONTENT_WIDTH} {total_height:.2f} re B"
        )
        
        text_y = self.y_cursor - pad_y
        for l in lines:
            text_y -= line_height
            if text_y < (MARGIN_BOTTOM + 10):
                self.y_cursor = text_y
                self.new_page()
                text_y = self.y_cursor - pad_y - line_height
            clean_l = escape_pdf_string(l)
            self.commands.append(f"BT /F4 {font_size} Tf 0.1 0.15 0.2 rg {MARGIN_LEFT + pad_x:.2f} {text_y:.2f} Td ({clean_l}) Tj ET")
            
        self.y_cursor = box_y - 8

    def draw_table(self, rows: list):
        if not rows:
            return
        # Parse table cells
        grid = []
        for r in rows:
            cells = [escape_pdf_string(c.strip()) for c in r.strip("|").split("|")]
            grid.append(cells)
            
        if len(grid) < 2:
            return
            
        # Ignore separator row (|---|---|)
        headers = grid[0]
        data_rows = [row for row in grid[1:] if not all(c.strip().startswith("-") or c.strip() == "" for c in row)]
        
        num_cols = len(headers)
        if num_cols == 0:
            return
            
        col_w = CONTENT_WIDTH / num_cols
        font_size = 7.5
        row_height = 16.0
        header_height = 18.0
        total_height = header_height + len(data_rows) * row_height
        
        self.ensure_space(min(total_height, 120))
        
        # Draw Header
        y_top = self.y_cursor
        self.commands.append(f"0.91 0.94 0.97 rg 0.7 0.77 0.85 RG 0.5 w {MARGIN_LEFT} {y_top - header_height:.2f} {CONTENT_WIDTH} {header_height:.2f} re B")
        for c_idx, h_text in enumerate(headers):
            x_cell = MARGIN_LEFT + c_idx * col_w + 4
            y_cell = y_top - 12
            self.commands.append(f"BT /F2 {font_size} Tf 0.05 0.12 0.25 rg {x_cell:.2f} {y_cell:.2f} Td ({h_text[:30]}) Tj ET")
            
        self.y_cursor -= header_height
        
        # Draw Data Rows
        for r_idx, row in enumerate(data_rows):
            self.ensure_space(row_height)
            bg_color = "0.98 0.99 1.0" if r_idx % 2 == 1 else "1.0 1.0 1.0"
            y_row = self.y_cursor
            self.commands.append(f"{bg_color} rg 0.8 0.85 0.9 RG 0.5 w {MARGIN_LEFT} {y_row - row_height:.2f} {CONTENT_WIDTH} {row_height:.2f} re B")
            for c_idx, val in enumerate(row):
                if c_idx >= num_cols:
                    break
                x_cell = MARGIN_LEFT + c_idx * col_w + 4
                y_cell = y_row - 11
                self.commands.append(f"BT /F1 {font_size} Tf 0.15 0.15 0.15 rg {x_cell:.2f} {y_cell:.2f} Td ({val[:35]}) Tj ET")
            self.y_cursor -= row_height
            
        self.y_cursor -= 10

    def draw_hr(self):
        self.ensure_space(15)
        self.y_cursor -= 6
        self.commands.append(f"0.8 0.84 0.88 RG 0.5 w {MARGIN_LEFT} {self.y_cursor:.2f} m {PAGE_WIDTH - MARGIN_RIGHT} {self.y_cursor:.2f} l S")
        self.y_cursor -= 8

    def finish(self):
        if self.commands:
            self.draw_page_decorations()
            stream = "\n".join(self.commands).encode("latin-1", errors="replace")
            self.doc.pages.append(stream)

def compile_text_to_pdf(txt_path: Path, pdf_path: Path):
    print(f"Reading {txt_path}...")
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    
    doc = PDFDocument()
    writer = FlowableWriter(doc)
    
    in_code = False
    code_lines = []
    in_table = False
    table_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Code fence
        if stripped.startswith("```"):
            if in_code:
                writer.draw_code_block(code_lines)
                code_lines = []
                in_code = False
            else:
                if in_table:
                    writer.draw_table(table_lines)
                    table_lines = []
                    in_table = False
                in_code = True
            i += 1
            continue
            
        if in_code:
            code_lines.append(line)
            i += 1
            continue
            
        # Tables
        if "|" in line and (line.startswith("|") or stripped.endswith("|")):
            in_table = True
            table_lines.append(stripped)
            i += 1
            continue
        elif in_table:
            writer.draw_table(table_lines)
            table_lines = []
            in_table = False
            
        if not stripped:
            i += 1
            continue
            
        # Horizontal rules
        if stripped in ("---", "***", "___") or re.match(r'^-{3,}$', stripped):
            writer.draw_hr()
            i += 1
            continue
            
        # Headings
        if stripped.startswith("#"):
            m = re.match(r'^(#{1,6})\s+(.*)$', stripped)
            if m:
                level = len(m.group(1))
                h_text = m.group(2)
                # Remove bold markers inside heading if present
                h_text = re.sub(r'\*\*(.*?)\*\*', r'\1', h_text)
                writer.draw_heading(h_text, level)
                i += 1
                continue
                
        # Bullet list items
        if re.match(r'^[-*]\s+(.*)$', stripped):
            item_text = re.match(r'^[-*]\s+(.*)$', stripped).group(1)
            # Remove bold/italic tags for smooth rendering
            item_text = re.sub(r'\*\*(.*?)\*\*', r'\1', item_text)
            writer.draw_paragraph("*  " + item_text, font_size=9.0, line_height=12.5, indent=12.0)
            i += 1
            continue
            
        # Numbered list items
        if re.match(r'^\d+\.\s+(.*)$', stripped):
            item_text = stripped
            item_text = re.sub(r'\*\*(.*?)\*\*', r'\1', item_text)
            writer.draw_paragraph(item_text, font_size=9.0, line_height=12.5, indent=12.0)
            i += 1
            continue
            
        # Normal paragraphs
        clean_para = re.sub(r'\*\*(.*?)\*\*', r'\1', stripped)
        clean_para = re.sub(r'`(.*?)`', r'\1', clean_para)
        writer.draw_paragraph(clean_para, font_size=9.2, line_height=13.0)
        i += 1

    if in_code:
        writer.draw_code_block(code_lines)
    if in_table:
        writer.draw_table(table_lines)
        
    writer.finish()
    print(f"Compiling PDF stream across {len(doc.pages)} pages...")
    doc.write_pdf(pdf_path)
    print(f"Successfully generated: {pdf_path.resolve()} ({pdf_path.stat().st_size} bytes)")

if __name__ == "__main__":
    src = Path("ZERO_TEXT_RESEARCH_PAPER.txt")
    out = Path("ZERO_TEXT_RESEARCH_PAPER.pdf")
    compile_text_to_pdf(src, out)
