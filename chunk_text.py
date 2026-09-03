from pypdf import PdfReader

reader = PdfReader("progit.pdf")

text = ""
for page_num in range(15, 65):   # skip pages 0-14 (TOC/license/preface), take next 50 real pages
    text += reader.pages[page_num].extract_text()

def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

chunks = chunk_text(text)

print(f"Total chunks created: {len(chunks)}")
print("---- Preview of chunk 0 ----")
print(chunks[0])
print("---- Preview of chunk 1 ----")
print(chunks[1])