# -*- coding: utf-8 -*-
"""
scripts/make_survey_pdf.py
==========================
Biên dịch file BAO_CAO_SURVEY_BENCHMARK_DACNTT.md thành file PDF học thuật chuẩn mực.
"""

import os
import sys
import subprocess
import markdown_it

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MD_PATH = os.path.join(ROOT, "BAO_CAO_SURVEY_BENCHMARK_DACNTT.md")

OUTPUTS = [
    os.path.join(ROOT, "BAO_CAO_SURVEY_BENCHMARK_DACNTT.pdf"),
    os.path.join(os.path.dirname(ROOT), "BAO_CAO_SURVEY_BENCHMARK_DACNTT.pdf"),
]

CSS = """
@page {
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
    @bottom-center {
        content: counter(page);
        font-size: 9.5pt;
        color: #64748b;
        font-family: 'Times New Roman', Times, serif;
    }
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.6;
    color: #1e293b;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
}

h1 {
    font-size: 16pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2.5px solid #1d4ed8;
    padding-bottom: 6px;
    margin-top: 22pt;
    margin-bottom: 12pt;
    page-break-after: avoid;
    text-transform: uppercase;
}

h2 {
    font-size: 13pt;
    font-weight: 700;
    color: #1e3a8a;
    border-bottom: 1.5px solid #cbd5e1;
    padding-bottom: 4px;
    margin-top: 18pt;
    margin-bottom: 8pt;
    page-break-after: avoid;
}

h3 {
    font-size: 11pt;
    font-weight: 600;
    color: #1d4ed8;
    margin-top: 14pt;
    margin-bottom: 6pt;
    page-break-after: avoid;
}

h4 {
    font-size: 10.5pt;
    font-weight: 600;
    color: #334155;
    margin-top: 10pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
}

p {
    margin-top: 0;
    margin-bottom: 8pt;
    text-align: justify;
}

ul, ol {
    margin-top: 0;
    margin-bottom: 8pt;
    padding-left: 20px;
}

li {
    margin-bottom: 3pt;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 8pt;
    margin-bottom: 12pt;
    font-size: 8.5pt;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #cbd5e1;
    padding: 5px 8px;
    text-align: left;
    vertical-align: top;
}

th {
    background-color: #f1f5f9;
    font-weight: 600;
    color: #0f172a;
}

tr:nth-child(even) {
    background-color: #f8fafc;
}

blockquote {
    border-left: 3.5px solid #3b82f6;
    background-color: #eff6ff;
    padding: 8px 12px;
    margin: 8pt 0;
    color: #1e40af;
    font-size: 9pt;
    border-radius: 0 4px 4px 0;
    page-break-inside: avoid;
}

blockquote p {
    margin: 0;
}

pre, code {
    font-family: "Cascadia Code", "Fira Code", Consolas, "Courier New", monospace;
}

code {
    background-color: #f1f5f9;
    color: #b91c1c;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 8.5pt;
}

pre {
    background-color: #0f172a;
    color: #f8fafc;
    padding: 10px 14px;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 8pt;
    line-height: 1.4;
    margin-top: 6pt;
    margin-bottom: 10pt;
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
    margin: 14pt 0;
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
    <title>Báo Cáo Nghiên Cứu Tổng Quan & Đối Sánh Thực Nghiệm (DACNTT)</title>
    <style>{CSS}</style>
</head>
<body>
{body_html}
</body>
</html>
"""
    temp_html = os.path.join(ROOT, "temp_survey.html")
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
