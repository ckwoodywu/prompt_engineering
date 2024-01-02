import gc

from flask import Flask, render_template, request, jsonify
from langchain.vectorstores.faiss import FAISS
from transformers import AutoTokenizer
from werkzeug.utils import secure_filename
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import openai
import os
from datetime import datetime
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceInstructEmbeddings

from huggingface_hub import hf_hub_download

from langchain.chains import RetrievalQA

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

from langchain.text_splitter import CharacterTextSplitter

from langchain import OpenAI, VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Milvus
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

# Milvus Database
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)

from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from transformers import AutoTokenizer
from langchain import HuggingFacePipeline
import transformers
import torch

app = Flask(__name__)
# Set openai key in env
openai.api_key = os.environ["OPENAI_API_KEY"] = "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"

# File upload configuration
# Save all uploaded files in the "uploads/"
app.config['UPLOAD_FOLDER'] = 'uploads/'
# Allow these fomat
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'json', 'csv', 'pdf', 'docx'])

# Milvus Database Collection name
COLLECTION_NAME = "pdf_collection"

# Remove Old Milvus collection
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
# milvus_connection = connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)


# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


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
model = "meta-llama/Llama-2-13b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    # max_length=1000,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1
)
chat_history=[]
def llama_response(directory, query):
    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    #
    # model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
    # model_basename = "llama-2-13b-chat.Q5_K_M.gguf"
    # model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
    #
    # llm = LlamaCpp(
    #     model_path=model_path,
    #     n_ctx=6000,
    #     n_gpu_layers=512,
    #     n_batch=30,
    #     callback_manager=callback_manager,
    #     max_tokens=4095,
    #     # max_tokens=256,
    #     n_parts=1,
    # )

    # llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})
    llm = HuggingFacePipeline(pipeline=pipeline)
    # load files from directory
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    # embedding engine
    hf_embedding = HuggingFaceInstructEmbeddings()

    db = FAISS.from_documents(docs, hf_embedding)

    # # save embeddings in local directory
    # db.save_local("uploads")
    #
    # # load from local
    # db = FAISS.load_local("uploads/", embeddings=hf_embedding)

    # use_original_text = request.form.get('useOriginalText') == 'on'
    # print(use_original_text)
    # if use_original_text:
    #     query = query + ", Just give me the original text you found."
    # else:
    #     query = query

    # docsearch = db.similarity_search(query)

    # chain = load_qa_chain(llm, chain_type="stuff", verbose=False)
    #
    # answer = chain.run(input_documents=search, question=query)
    # return answer
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(),
    #                                  return_source_documents=True)

    #   qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch, return_source_documents=True)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

    condense_q_system_prompt = """Given a chat history and the user question \
     which might reference the chat history, formulate a standalone question \
     which can be understood without the chat history. Do NOT answer the question, \
     just reformulate it if needed and otherwise return it as is."""

    condense_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    condense_q_chain = condense_q_prompt | llm | StrOutputParser()

    # condense_q_chain.invoke(
    #     {
    #         "chat_history": [
    #             HumanMessage(content="What does LLM stand for?"),
    #             AIMessage(content="Large language model"),
    #         ],
    #         "question": "What is meant by large",
    #     }
    # )
    #
    # condense_q_chain.invoke(
    #     {
    #         "chat_history": [
    #             HumanMessage(content="What does LLM stand for?"),
    #             AIMessage(content="Large language model"),
    #         ],
    #         "question": "How do transformers work",
    #     }
    # )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
     Use the following pieces of retrieved context to answer the question. \
     If you don't know the answer, just say that you don't know. \
     Use three sentences maximum and keep the answer concise.\

     {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    def condense_question(input: dict):
        print("Input content:", input)
        if input.get("chat_history"):
            return condense_q_chain
        else:
            return input["question"]

    rag_chain = (
            RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
            | qa_prompt
            | llm
    )
    question = query
    # Invoke the RAG chain and get the AI message
    ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
    # print('ai_msg: ', ai_msg.content)
    # Extract the content from the AIMessage object
    # ai_msg_content = ai_msg.content

    # Append the human message and the AI's response content to the chat history
    chat_history.extend([HumanMessage(content=question), AIMessage(content=ai_msg)])

    # Return only the content of the AI's response
    return ai_msg

chat_history = []
def rag_mode(directory, query):
    # Remove Old Milvus collection
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"
    milvus_connection = connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    docsearch = ""

    # Creating new Vector DB
    if not utility.has_collection(COLLECTION_NAME):
        loader = DirectoryLoader(directory)

        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        split_docs = text_splitter.split_documents(documents)

        new_doc = []
        for doc in split_docs:
            met = doc.metadata
            met['title'] = "L"
            met['description'] = "L"
            met['language'] = 'us'
            new_doc.append(Document(page_content=doc.page_content, metadata=met))
        #    	continue

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        embeddings = OpenAIEmbeddings()

        # Create New Milvus collection & Store new documents into the collection
        docsearch = create_collection(new_doc, embeddings)

    else:

        loader = DirectoryLoader(directory)

        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        split_docs = text_splitter.split_documents(documents)

        new_doc = []
        for doc in split_docs:
            met = doc.metadata
            met['title'] = "L"
            met['description'] = "L"
            met['language'] = 'us'
            new_doc.append(Document(page_content=doc.page_content, metadata=met))
        #    	continue

        # Loading Vector DB

        collection = Collection(COLLECTION_NAME)  # Get an existing collection.

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        embeddings = OpenAIEmbeddings()

        docsearch = load_collection(new_doc, embeddings)

    # Create New Milvus collection & Store new documents into the collection
    #    docsearch = Milvus.from_documents(
    #	new_doc,
    #	embedding=embeddings,
    #	collection_name=COLLECTION_NAME,
    #	connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
    #    )

    # Create an Index for the new Milvus collection
    #    index_param = {
    #        'index_type': "IVF_FLAT",
    #        'params': {'nlist': 16384},  # Adjust 'nlist' based on your dataset size and characteristics
    #        'metric_type': "L2"  # Choose the appropriate metric type for your vectors
    #    }

    #    collection = Collection(COLLECTION_NAME)
    #    collection.create_index(
    #    	field_name="text",
    #  	index_params=index_param)

    #    print(f"Index created for collection: {collection_name}")

    #    utility.index_building_progress(COLLECTION_NAME)

    #    docsearch = Chroma.from_documents(split_docs, embeddings)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", callback_manager=callback_manager)

    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(),
    #                                  return_source_documents=True)

    #   qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch, return_source_documents=True)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

    condense_q_system_prompt = """Given a chat history and the latest user question \
    which might reference the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    condense_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    condense_q_chain = condense_q_prompt | llm | StrOutputParser()

    condense_q_chain.invoke(
        {
            "chat_history": [
                HumanMessage(content="What does LLM stand for?"),
                AIMessage(content="Large language model"),
            ],
            "question": "What is meant by large",
        }
    )

    condense_q_chain.invoke(
        {
            "chat_history": [
                HumanMessage(content="What does LLM stand for?"),
                AIMessage(content="Large language model"),
            ],
            "question": "How do transformers work",
        }
    )

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    def condense_question(input: dict):
        if input.get("chat_history"):
            return condense_q_chain
        else:
            return input["question"]

    rag_chain = (
            RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
            | qa_prompt
            | llm
    )

    question = query
    # Invoke the RAG chain and get the AI message
    ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})

    # Extract the content from the AIMessage object
    ai_msg_content = ai_msg.content

    # Append the human message and the AI's response content to the chat history
    chat_history.extend([HumanMessage(content=question), ai_msg])

    # Return only the content of the AI's response
    return ai_msg_content



# Chatbot function
def gpt3chatbot(message):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    chat_message = response['choices'][0]['message']['content']
    messages.append({"role": "assistant", "content": chat_message})
    return chat_message


def llamachatbot(message):
    model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
    model_basename = "llama-2-13b-chat.Q5_K_M.gguf"
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # load the large language model file
    # LLM = Llama(model_path="llama-2-7b.ggmlv3.q2_K.bin")
    # LLM = Llama(model_path=model_path)
    LLM = LlamaCpp(
        model_path=model_path,
        n_ctx=6000,
        n_gpu_layers=512,
        n_batch=30,
        callback_manager=callback_manager,
        max_tokens=4095,
        n_parts=1,
    )
    # create a text prompt
    prompt = message
    print("Starting to generate llama2-rag-off response...")
    # generate a response (takes several seconds)
    output = LLM(prompt)
    print("Response generated.")
    return output


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('user_input', '')
    mode = request.form.get('mode', 'online')
    model = request.form.get('model', 'gpt-3.5')
    rag = request.form.get('rag', 'off')
    use_entire_uploads = 'useEntireUploads' in request.form
    bot_response = ""  # Initialize bot_response to an empty string or a default value

    if mode == 'offline' and model == 'llama' and rag == 'on':
        files = request.files.getlist('directory')
        if not files:
            return jsonify({'response': "No files uploaded."})

        if use_entire_uploads:
            directory_path = app.config['UPLOAD_FOLDER']
            bot_response = llama_response(directory_path, user_input)
            return jsonify({'response': bot_response})

        else:
            valid_files = [file for file in files if file and allowed_file(file.filename)]
            invalid_files = [file.filename for file in files if file and not allowed_file(file.filename)]

            if invalid_files:
                return jsonify({'response': f"Invalid file type."})

            if valid_files:
                # Create a new directory named with timestamp
                current_time = datetime.now().strftime("%Y%m%d%H%M%S")
                directory_name = os.path.splitext(files[0].filename)[0] + "_" + current_time
                directory_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(directory_name))
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)

                # Save uploaded files to new directory
                for file in valid_files:
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(directory_path, filename))

                bot_response = llama_response(directory_path, user_input)
                return jsonify({'response': bot_response})

    elif mode == 'online' and model == 'gpt-3.5' and rag == 'on':
        files = request.files.getlist('directory')
        if not files:
            return jsonify({'response': "No files uploaded."})

        if use_entire_uploads:
            directory_path = app.config['UPLOAD_FOLDER']
            bot_response = rag_mode(directory_path, user_input)
            return jsonify({'response': bot_response})
        else:
            valid_files = [file for file in files if file and allowed_file(file.filename)]
            invalid_files = [file.filename for file in files if file and not allowed_file(file.filename)]

            if invalid_files:
                return jsonify({'response': f"Invalid file type."})

            if valid_files:
                # Create a new directory named with timestamp
                current_time = datetime.now().strftime("%Y%m%d%H%M%S")
                directory_name = os.path.splitext(files[0].filename)[0] + "_" + current_time
                directory_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(directory_name))
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)

                # Save uploaded files to new directory
                for file in valid_files:
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(directory_path, filename))
                bot_response = rag_mode(directory_path, user_input)
                return jsonify({'response': bot_response})
            else:
                return jsonify({'response': "No valid files uploaded."})
    elif mode == 'online' and model == 'gpt-3.5' and rag == 'off':
        bot_response = gpt3chatbot(user_input)
        return jsonify({'response': bot_response})

    elif mode == 'offline' and model == 'llama' and rag == 'off':
        bot_response = llamachatbot(user_input)

    return jsonify({'response': bot_response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8006)
