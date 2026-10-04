import os

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

KNOWLEDGE_BASE_DIR = os.path.join(
    BASE_DIR,
    "knowledge_base"
)

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ==========================================
# INITIALIZE CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_or_create_collection(
    name="healthcare_knowledge"
)

print("ChromaDB initialized successfully.")


# ==========================================
# CHUNKING FUNCTION
# ==========================================

def create_chunks(text, source):
    """
    Split a markdown knowledge file into sections.

    Each section beginning with '## ' becomes one
    knowledge chunk.
    """

    sections = text.split("\n## ")

    chunks = []

    for section in sections:

        section = section.strip()

        if not section:
            continue

        # Restore heading removed by split()
        if not section.startswith("# "):
            section = "## " + section

        chunks.append(
            {
                "text": section,
                "source": source
            }
        )

    return chunks


# ==========================================
# BUILD KNOWLEDGE BASE
# ==========================================

def build_knowledge_base():
    """
    Read all Markdown files, create embeddings,
    clear the existing ChromaDB collection, and
    store the new knowledge chunks.
    """

    # --------------------------------------
    # Check knowledge base directory
    # --------------------------------------

    if not os.path.isdir(KNOWLEDGE_BASE_DIR):

        raise FileNotFoundError(
            f"Knowledge base directory not found: "
            f"{KNOWLEDGE_BASE_DIR}"
        )

    # --------------------------------------
    # Clear old collection
    # --------------------------------------

    existing = collection.get()

    if existing["ids"]:

        collection.delete(
            ids=existing["ids"]
        )

    print("Old knowledge chunks cleared.")

    # --------------------------------------
    # Read and chunk knowledge files
    # --------------------------------------

    all_chunks = []

    for filename in sorted(
        os.listdir(KNOWLEDGE_BASE_DIR)
    ):

        if not filename.endswith(".md"):
            continue

        file_path = os.path.join(
            KNOWLEDGE_BASE_DIR,
            filename
        )

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        chunks = create_chunks(
            text,
            filename
        )

        all_chunks.extend(chunks)

        print(
            f"Loaded {filename}: "
            f"{len(chunks)} chunks"
        )

    print(
        "\nTotal knowledge chunks:",
        len(all_chunks)
    )

    if not all_chunks:

        raise ValueError(
            "No knowledge chunks were found."
        )

    # --------------------------------------
    # Prepare data
    # --------------------------------------

    documents = [
        chunk["text"]
        for chunk in all_chunks
    ]

    metadatas = [
        {
            "source": chunk["source"]
        }
        for chunk in all_chunks
    ]

    ids = [
        f"chunk_{i}"
        for i in range(len(all_chunks))
    ]

    # --------------------------------------
    # Create embeddings
    # --------------------------------------

    print("\nCreating embeddings...")

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True
    ).tolist()

    print(
        "Embeddings created successfully."
    )

    # --------------------------------------
    # Store in ChromaDB
    # --------------------------------------

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        "\nKnowledge base stored successfully."
    )

    print(
        "Total chunks in ChromaDB:",
        collection.count()
    )

    return collection.count()


# ==========================================
# DEPLOYMENT-SAFE INITIALIZATION
# ==========================================

def ensure_knowledge_base():
    """
    Make sure the ChromaDB knowledge collection
    exists and contains knowledge chunks.

    On a fresh deployment, ChromaDB may be empty.
    In that case, automatically build the knowledge
    base from the tracked Markdown files.
    """

    current_count = collection.count()

    if current_count > 0:

        print(
            "Knowledge base already initialized."
        )

        return current_count

    print(
        "Knowledge base is empty."
    )

    print(
        "Building knowledge base from Markdown files..."
    )

    return build_knowledge_base()


# ==========================================
# RETRIEVE KNOWLEDGE
# ==========================================

def retrieve_knowledge(
    query,
    n_results=3
):
    """
    Retrieve the most relevant knowledge sections
    for a given query.

    Returns a list of dictionaries containing:
    - text
    - source
    """

    if not query or not query.strip():

        return []

    # Make sure a fresh deployment has
    # an initialized knowledge base.
    ensure_knowledge_base()

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    # Do not request more results than
    # are available in the collection.
    result_count = min(
        n_results,
        collection.count()
    )

    if result_count <= 0:

        return []

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=result_count
    )

    retrieved = []

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    for i, document in enumerate(
        documents
    ):

        source = (
            metadatas[i]["source"]
            if i < len(metadatas)
            else "unknown"
        )

        retrieved.append(
            {
                "text": document,
                "source": source
            }
        )

    return retrieved


# ==========================================
# SIMPLE EVIDENCE RETRIEVAL
# ==========================================

def get_evidence(
    query,
    n_results=3
):
    """
    Retrieve knowledge that can be displayed
    as supporting evidence in the dashboard.
    """

    results = retrieve_knowledge(
        query=query,
        n_results=n_results
    )

    evidence = []

    for result in results:

        evidence.append(
            {
                "source": result["source"],
                "text": result["text"]
            }
        )

    return evidence


# ==========================================
# RETRIEVAL TEST
# ==========================================

def run_retrieval_tests():

    queries = [

        (
            "Why is a patient's current reading "
            "compared with their personal baseline?"
        ),

        (
            "What does it mean when heart rate, "
            "SpO2, pulse rate and respiratory rate "
            "change together?"
        ),

        (
            "Why should an abnormal sensor pattern "
            "be reviewed by a human?"
        )
    ]

    for query in queries:

        print("\n")
        print("=" * 70)

        print("QUERY:")

        print(query)

        print("=" * 70)

        results = retrieve_knowledge(
            query=query,
            n_results=2
        )

        for i, result in enumerate(
            results
        ):

            print(
                f"\nRESULT {i + 1}"
            )

            print(
                "Source:",
                result["source"]
            )

            print(
                "\nRetrieved section:"
            )

            print(
                result["text"]
            )

            print(
                "\n" + "-" * 60
            )


# ==========================================
# INITIALIZE KNOWLEDGE BASE ON IMPORT
# ==========================================

ensure_knowledge_base()


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print(
        "\nRunning RAG retrieval tests..."
    )

    run_retrieval_tests()

    print(
        "\nRAG retrieval test completed successfully."
    )