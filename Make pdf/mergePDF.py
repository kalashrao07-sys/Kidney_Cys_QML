from pypdf import PdfWriter, PdfReader

# List your two PDF files here (in the order you want them merged)
pdf_files = ["file1.pdf", "file2.pdf"]

writer = PdfWriter()

for pdf_file in pdf_files:
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        writer.add_page(page)

with open("Plagiarism Report_430.pdf", "wb") as output:
    writer.write(output)

print("Merged PDF saved as 'Plagiarism Report_430.pdf'")