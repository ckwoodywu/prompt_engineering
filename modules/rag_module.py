# Import Libraries
from langchain.vectorstores import Milvus
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings


def vectordb_embedding(method):
    if method.upper() == "OPENAI":
        embeddings = OpenAIEmbeddings()
    
    elif method.upper() == "HUGGINGFACE":
        embeddings = HuggingFaceInstructEmbeddings()

    else:
        print(f"Caution : Unknown Embedding Method <{method}>. OpenAIEmbeddings is thus Used by Default")
        embeddings = OpenAIEmbeddings()

    return embeddings

# Create new Vector DB
def init_vectordb(directory, 
                  textsplit_chunk_size,
                  textsplit_chunk_overlap,
                  search_type="similarity",
                  search_kwargs={"k":6},
                  embedding_method="OpenAI"):

    # Remove Old Milvus collection
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"
    milvus_connection = connections.connect("default", 
                                            host=MILVUS_HOST, 
                                            port=MILVUS_PORT)
 
    loader = DirectoryLoader(directory)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=textsplit_chunk_size, 
                                          chunk_overlap=textsplit_chunk_overlap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=textsplit_chunk_size,
                                                   chunk_overlap=textsplit_chunk_overlap)

    split_docs = text_splitter.split_documents(documents)

    new_doc = []
    for doc in split_docs:
        met = doc.metadata
        met['title'] = "L"
        met['description'] = "L"
        met['language'] = 'us'
        new_doc.append(Document(page_content=doc.page_content, metadata=met))
    #    	continue

    embeddings = vectordb_embedding(embedding_method)

    docsearch = ""
    # Create New Milvus collection & Store new documents into the collection
    if not utility.has_collection(COLLECTION_NAME):
        docsearch = create_collection(new_doc, embeddings)

    # Retrieve Existing Vector Store
    else:
        """ *** Refine Later >>> Where would this parameter be used? *** """
        collection = Collection(COLLECTION_NAME)  # Get an existing collection
        docsearch = load_collection(new_doc, embeddings)

    retriever = docsearch.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    return retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Milvus Database
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)

# Milvus Database Collection name
COLLECTION_NAME = "pdf_collection"

# Remove Old Milvus collection
""" *** Refine Later >>> Need to convert as parameters *** """
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
# milvus_connection = connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

def create_collection(new_doc, embedding):
    milvusDb = Milvus.from_documents(
        new_doc,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
    )
    return milvusDb

def load_collection(new_doc, embedding):
    milvusDb = Milvus.from_documents(
        new_doc,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
    )

#    milvusDb = Milvus(
#        embeddings,
#        collection_name=COLLECTION_NAME,
#        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
#    )
    return milvusDb

