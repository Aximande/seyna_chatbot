from PyPDF2 import PdfReader

reader = PdfReader("documents/ski.pdf")
page = reader.pages[0]
print(page.extract_text())
