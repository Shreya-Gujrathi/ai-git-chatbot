from pypdf import PdfReader


reader = PdfReader("progit.pdf")
page = reader.pages[15]
print(page.extract_text()[:800])



# Let's extract just the first 50 pages for now (covers early chapters)
text = ""
for page_num in range(50):
    page = reader.pages[page_num]
    text += page.extract_text()

print(f"Total characters extracted: {len(text)}")
print("---- Preview of first 500 characters ----")
print(text[:500])