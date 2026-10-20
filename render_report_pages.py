from pathlib import Path
import fitz
pdf=fitz.open('report_render/report.pdf')
for i,page in enumerate(pdf):
    page.get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(f'report_render/page-{i+1}.png')
    print(i+1, len(page.get_text()), page.get_text()[:85].replace('\n',' / '))
