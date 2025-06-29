import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text



def chunk_text(text, max_sentences=4):
    # Split text into sentences using regex
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = ' '.join(sentences[i:i+max_sentences]).strip()
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

