import re
import html
from pathlib import Path

def convert_markdown_to_html(md_text: str) -> str:
    lines = md_text.splitlines()
    html_lines = []
    in_code_block = False
    in_table = False
    table_rows = []
    in_ul = False
    in_ol = False
    
    def close_lists():
        nonlocal in_ul, in_ol
        res = []
        if in_ul:
            res.append("</ul>")
            in_ul = False
        if in_ol:
            res.append("</ol>")
            in_ol = False
        return res

    def close_table():
        nonlocal in_table, table_rows
        if not in_table:
            return []
        res = ["<div class='table-container'><table>"]
        # Determine header vs body
        if len(table_rows) >= 2 and all(c in "-: |" for c in table_rows[1].strip()):
            # Header row
            headers = [c.strip() for c in table_rows[0].strip("|").split("|")]
            res.append("<thead><tr>")
            for h in headers:
                res.append(f"<th>{format_inline(h)}</th>")
            res.append("</tr></thead>")
            res.append("<tbody>")
            for row in table_rows[2:]:
                cells = [c.strip() for c in row.strip("|").split("|")]
                res.append("<tr>")
                for cell in cells:
                    res.append(f"<td>{format_inline(cell)}</td>")
                res.append("</tr>")
            res.append("</tbody>")
        else:
            res.append("<tbody>")
            for row in table_rows:
                cells = [c.strip() for c in row.strip("|").split("|")]
                res.append("<tr>")
                for cell in cells:
                    res.append(f"<td>{format_inline(cell)}</td>")
                res.append("</tr>")
            res.append("</tbody>")
        res.append("</table></div>")
        in_table = False
        table_rows = []
        return res

    def format_inline(text: str) -> str:
        # Escape html entities first (except if already converted)
        text = html.escape(text)
        # Bold + Italic: ***text***
        text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', text)
        # Bold: **text**
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        # Italic: *text*
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        # Inline code: `text`
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
        return text

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Code block fence
        if stripped.startswith("```"):
            html_lines.extend(close_lists())
            html_lines.extend(close_table())
            if in_code_block:
                html_lines.append("</code></pre>")
                in_code_block = False
            else:
                lang = stripped[3:].strip()
                html_lines.append(f"<pre><code class='language-{lang}'>")
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            html_lines.append(html.escape(line))
            i += 1
            continue

        # Tables
        if "|" in line and (line.startswith("|") or stripped.endswith("|")):
            html_lines.extend(close_lists())
            in_table = True
            table_rows.append(stripped)
            i += 1
            continue
        elif in_table:
            html_lines.extend(close_table())

        # Empty line
        if not stripped:
            html_lines.extend(close_lists())
            i += 1
            continue

        # Horizontal rule
        if stripped in ("---", "***", "___") or re.match(r'^-{3,}$', stripped):
            html_lines.extend(close_lists())
            html_lines.append("<hr/>")
            i += 1
            continue

        # Headings
        if stripped.startswith("#"):
            html_lines.extend(close_lists())
            m = re.match(r'^(#{1,6})\s+(.*)$', stripped)
            if m:
                level = len(m.group(1))
                h_text = format_inline(m.group(2))
                html_lines.append(f"<h{level}>{h_text}</h{level}>")
                i += 1
                continue

        # Unordered list: - or *
        if re.match(r'^[-*]\s+(.*)$', stripped):
            if in_ol:
                html_lines.extend(close_lists())
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            item_text = re.match(r'^[-*]\s+(.*)$', stripped).group(1)
            html_lines.append(f"<li>{format_inline(item_text)}</li>")
            i += 1
            continue

        # Ordered list: 1. or 2.
        if re.match(r'^\d+\.\s+(.*)$', stripped):
            if in_ul:
                html_lines.extend(close_lists())
            if not in_ol:
                html_lines.append("<ol>")
                in_ol = True
            item_text = re.match(r'^\d+\.\s+(.*)$', stripped).group(1)
            html_lines.append(f"<li>{format_inline(item_text)}</li>")
            i += 1
            continue

        # Normal paragraph
        html_lines.extend(close_lists())
        html_lines.append(f"<p>{format_inline(stripped)}</p>")
        i += 1

    html_lines.extend(close_lists())
    html_lines.extend(close_table())
    return "\n".join(html_lines)


def build_full_html_document(body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ZERO-TEXT Research Paper</title>
<style>
    @page {{
        size: A4;
        margin: 20mm 15mm 20mm 15mm;
        @bottom-right {{
            content: counter(page);
        }}
    }}
    body {{
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, 'Helvetica Neue', Arial, sans-serif;
        font-size: 10pt;
        line-height: 1.5;
        color: #1a1a1a;
        background-color: #ffffff;
        margin: 0;
        padding: 0;
    }}
    h1 {{
        font-size: 19pt;
        line-height: 1.25;
        color: #0b1c3d;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 12px;
        font-weight: 700;
        border-bottom: 2px solid #0b1c3d;
        padding-bottom: 8px;
    }}
    h2 {{
        font-size: 13.5pt;
        color: #102a5c;
        margin-top: 24px;
        margin-bottom: 8px;
        font-weight: 700;
        border-bottom: 1px solid #c2cde0;
        padding-bottom: 4px;
        page-break-after: avoid;
    }}
    h3 {{
        font-size: 11pt;
        color: #1f3b73;
        margin-top: 16px;
        margin-bottom: 6px;
        font-weight: 600;
        page-break-after: avoid;
    }}
    h4 {{
        font-size: 10pt;
        color: #2b4985;
        margin-top: 12px;
        margin-bottom: 4px;
        font-weight: 600;
        page-break-after: avoid;
    }}
    p {{
        margin-top: 0;
        margin-bottom: 7px;
        text-align: justify;
    }}
    ul, ol {{
        margin-top: 2px;
        margin-bottom: 7px;
        padding-left: 20px;
    }}
    li {{
        margin-bottom: 3px;
    }}
    hr {{
        border: 0;
        border-top: 1px solid #d0d7de;
        margin: 16px 0;
    }}
    code {{
        font-family: 'Consolas', 'Courier New', monospace;
        background-color: #f3f5f8;
        padding: 2px 4px;
        border-radius: 3px;
        font-size: 9pt;
        color: #b31d28;
    }}
    pre {{
        background-color: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 5px;
        padding: 10px 12px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 8pt;
        line-height: 1.35;
        overflow-x: auto;
        white-space: pre;
        margin: 10px 0;
        page-break-inside: avoid;
    }}
    pre code {{
        background: none;
        padding: 0;
        border-radius: 0;
        color: #24292f;
    }}
    .table-container {{
        width: 100%;
        overflow-x: auto;
        margin: 12px 0;
        page-break-inside: avoid;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 8.5pt;
        line-height: 1.35;
        margin: 4px 0;
    }}
    th, td {{
        border: 1px solid #b8c4d4;
        padding: 5px 8px;
        text-align: left;
        vertical-align: top;
    }}
    th {{
        background-color: #eaf0f8;
        font-weight: 600;
        color: #0b1c3d;
    }}
    tr:nth-child(even) {{
        background-color: #f8fafc;
    }}
    strong {{
        font-weight: 600;
        color: #0d1e3d;
    }}
    em {{
        font-style: italic;
    }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

def main():
    txt_path = Path("ZERO_TEXT_RESEARCH_PAPER.txt")
    if not txt_path.exists():
        print("Error: ZERO_TEXT_RESEARCH_PAPER.txt not found!")
        return

    print("Reading ZERO_TEXT_RESEARCH_PAPER.txt...")
    md_text = txt_path.read_text(encoding="utf-8")
    
    print("Converting Markdown to academic HTML document...")
    body_html = convert_markdown_to_html(md_text)
    full_html = build_full_html_document(body_html)
    
    html_out = Path("ZERO_TEXT_RESEARCH_PAPER.html")
    html_out.write_text(full_html, encoding="utf-8")
    print(f"HTML successfully generated: {html_out.resolve()} ({html_out.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
