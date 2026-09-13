# -*- coding: utf-8 -*-
"""
scripts/make_guide_pdf.py
=========================
Biên dịch file HUONG_DAN_CHI_TIET_XAY_DUNG_ADAPTIVE_TGCN.md thành file PDF chuẩn học thuật.
"""

import os
import sys
import subprocess
import markdown_it

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MD_PATH = os.path.join(ROOT, "HUONG_DAN_CHI_TIET_XAY_DUNG_ADAPTIVE_TGCN.md")

OUTPUTS = [
    os.path.join(os.path.dirname(ROOT), "HUONG_DAN_CHI_TIET_XAY_DUNG_ADAPTIVE_TGCN.pdf"),
    os.path.join(ROOT, "HUONG_DAN_CHI_TIET_XAY_DUNG_ADAPTIVE_TGCN.pdf"),
]

CSS = """
@page {
    size: A4;
    margin: 22mm 18mm 22mm 18mm;
    @bottom-center {
        content: counter(page);
        font-size: 10pt;
        color: #64748b;
    }
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.65;
    color: #1e293b;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
}

h1 {
    font-size: 18pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2.5px solid #2563eb;
    padding-bottom: 8px;
    margin-top: 24pt;
    margin-bottom: 12pt;
    page-break-after: avoid;
}

h2 {
    font-size: 14pt;
    font-weight: 600;
    color: #1e3a8a;
    border-bottom: 1.5px solid #e2e8f0;
    padding-bottom: 5px;
    margin-top: 18pt;
    margin-bottom: 10pt;
    page-break-after: avoid;
}

h3 {
    font-size: 12pt;
    font-weight: 600;
    color: #1d4ed8;
    margin-top: 14pt;
    margin-bottom: 6pt;
    page-break-after: avoid;
}

p {
    margin-top: 0;
    margin-bottom: 10pt;
    text-align: justify;
}

ul, ol {
    margin-top: 0;
    margin-bottom: 10pt;
    padding-left: 22px;
}

li {
    margin-bottom: 4pt;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10pt;
    margin-bottom: 14pt;
    font-size: 9pt;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #cbd5e1;
    padding: 6px 10px;
    text-align: left;
}

th {
    background-color: #f1f5f9;
    font-weight: 600;
    color: #0f172a;
}

tr:nth-child(even) {
    background-color: #f8fafc;
}

pre, code {
    font-family: "Cascadia Code", "Fira Code", Consolas, "Courier New", monospace;
}

code {
    background-color: #f1f5f9;
    color: #b91c1c;
    padding: 2px 5px;
    border-radius: 4px;
    font-size: 9pt;
}

pre {
    background-color: #0f172a;
    color: #f8fafc;
    padding: 12px 16px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 8.5pt;
    line-height: 1.45;
    margin-top: 8pt;
    margin-bottom: 12pt;
    page-break-inside: avoid;
}

pre code {
    background-color: transparent;
    color: inherit;
    padding: 0;
    border-radius: 0;
    font-size: inherit;
}

hr {
    border: none;
    border-top: 1px solid #cbd5e1;
    margin: 16pt 0;
}
"""


def main():
    if not os.path.exists(MD_PATH):
        print(f"[ERROR] Not found: {MD_PATH}")
        return

    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    parser = markdown_it.MarkdownIt().enable("table").enable("strikethrough")
    body_html = parser.render(md_text)

    full_html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Hướng Dẫn Xây Dựng Mô Hình Adaptive T-GCN</title>
    <style>{CSS}</style>
</head>
<body>
{body_html}
</body>
</html>
"""
    temp_html = os.path.join(ROOT, "temp_guide.html")
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(full_html)

    chrome_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_exe):
        chrome_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

    print(f"Using browser: {chrome_exe}")
    for out_pdf in OUTPUTS:
        cmd = [
            chrome_exe,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={out_pdf}",
            temp_html,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(out_pdf):
            print(f"[SUCCESS] Exported: {out_pdf} ({os.path.getsize(out_pdf)/1024:.1f} KB)")
        else:
            print(f"[ERROR] Failed to export {out_pdf}")

    if os.path.exists(temp_html):
        os.remove(temp_html)


if __name__ == "__main__":
    main()
