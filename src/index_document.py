#Now we will work on the chunking and loading of the pdf 

import warnings

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

warnings.filterwarnings("ignore", category=DeprecationWarning)

console = Console()

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

# def main():
#     raw_docs = load_pdf()
#     print("PDF loaded successfully")
#     print(f"Number of PDF pages loaded : {len(raw_docs)}")

#     chunked_docs = create_chunks(raw_docs)
#     print("Chunking completed")
#     print(f"Number of chunks created: {len(chunked_docs)}")

#     print("\n First chunk preview: ")
#     print("-"*50)
#     print(chunked_docs[0].page_content[:1000])

#     print("\nFirst chunk metadata:")
#     print("-" * 50)
#     print(chunked_docs[0].metadata)

# if __name__ == "__main__":
#     main()

def main():
    console.print(
        Panel.fit(
            "[bold cyan]DSA PDF RAG - Document Indexing[/bold cyan]",
            border_style="cyan"
        )
    )

    raw_docs = load_pdf()
    chunked_docs = create_chunks(raw_docs)

    summary_table = Table(
        title="Indexing Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )

    summary_table.add_column("Task", style="cyan")
    summary_table.add_column("Status", justify="center")
    summary_table.add_column("Result", style="green")

    summary_table.add_row("PDF Loading", "✅", f"{len(raw_docs)} pages loaded")
    summary_table.add_row("Chunking", "✅", f"{len(chunked_docs)} chunks created")

    console.print(summary_table)

    console.print("\n[bold yellow]First Chunk Preview[/bold yellow]")
    console.print(
        Panel(
            chunked_docs[0].page_content[:1000],
            border_style="yellow"
        )
    )

    console.print("\n[bold yellow]First Chunk Metadata[/bold yellow]")

    metadata_table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold blue"
    )

    metadata_table.add_column("Key", style="cyan")
    metadata_table.add_column("Value", style="white")

    for key, value in chunked_docs[0].metadata.items():
        metadata_table.add_row(str(key), str(value))

    console.print(metadata_table)


if __name__ == "__main__":
    main()
