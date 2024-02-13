from PyPDF2 import PdfReader

reader = PdfReader(
    "documents/Accident de ski _ êtes-vous bien assuré _- Lafinancepour tous - 26 Janvier 2022.pdf"
)
page = reader.pages[0]
print(page.extract_text())
