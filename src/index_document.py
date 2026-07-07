#Now we will work on the chunking and loading of the pdf 



warnings.filterwarnings("ignore", category=DeprecationWarning)

console = Console()
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_Path = Path("/Users/nafisamehzabin/Desktop/Fun Coding Projects/dsa-pdf-rag-python/data/dsa.pdf")

def load_pdf():

    """
     This function loads the PDF.
     loader = PyPDFLoader(...)
        raw_docs = loader.load()
    """

    if not PDF_Path.exists():
        raise FileNotFoundError("Could not find data/dsa.pdf. Put your DSA PDF inside the data folder.")
    

    loader = PyPDFLoader(str(PDF_Path))
    raw_docs = loader.load()

    return raw_docs

def create_chunks(raw_docs):
    """
    This function breaks the PDF pages into smaller chunks.

    Why?
    Because we should not send the full PDF to the AI model.
    Instead, we split it into smaller searchable pieces.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= 1000,
        chunk_overlap=200,
    )

    chunked_docs= text_splitter.split_documents(raw_docs)

    return chunked_docs

def main():
    raw_docs = load_pdf()
    print("PDF loaded successfully")
    print(f"Number of PDF pages loaded : {len(raw_docs)}")

    chunked_docs = create_chunks(raw_docs)
    print("Chunking completed")
    print(f"Number of chunks created: {len(chunked_docs)}")

    print("\n First chunk preview: ")
    print("-"*50)
    print(chunked_docs[0].page_content[:1000])

    print("\nFirst chunk metadata:")
    print("-" * 50)
    print(chunked_docs[0].metadata)

if __name__ == "__main__":
    main()

