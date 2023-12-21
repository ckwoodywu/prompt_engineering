from flask import Flask, render_template, request, jsonify
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.faiss import FAISS
from llama_cpp import Llama
from werkzeug.utils import secure_filename
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings, LlamaCppEmbeddings
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

from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI,VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings
import torch

from chromadb.errors import InvalidDimensionException

from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate
)

from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain

from langchain.embeddings import HuggingFaceEmbeddings

from torch import cuda, bfloat16
import transformers

app = Flask(__name__)
# Set openai key in env
openai.api_key = os.environ["OPENAI_API_KEY"] = "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"

# File upload configuration
# Save all uploaded files in the "uploads/"
app.config['UPLOAD_FOLDER'] = 'uploads/'
# Allow these fomat
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'json', 'csv', 'pdf', 'docx'])

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def llama_response(directory, query):
#     callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#
#     model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
#     model_basename = "llama-2-13b-chat.Q5_K_M.gguf"
#     model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
#
#     llm = LlamaCpp(
#         model_path=model_path,
#         n_ctx=6000,
#         n_gpu_layers=512,
#         n_batch=30,
#         callback_manager=callback_manager,
#         max_tokens=4095,
#         # max_tokens=256,
#         n_parts=1,
#     )
#
#     # load files from directory
#     loader = DirectoryLoader(directory)
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
#     docs = text_splitter.split_documents(documents)
#
#     # embedding engine
#     hf_embedding = HuggingFaceInstructEmbeddings()
#
#     db = FAISS.from_documents(docs, hf_embedding)
#
#     # save embeddings in local directory
#     db.save_local("uploads")
#
#     # load from local
#     db = FAISS.load_local("uploads/", embeddings=hf_embedding)
#
#     use_original_text = request.form.get('useOriginalText') == 'on'
#     print(use_original_text)
#     if use_original_text:
#         query = query + "You must give me the original text you found."
#     else:
#         query = query
#
#     search = db.similarity_search(query)
#
#     chain = load_qa_chain(llm, chain_type="stuff", verbose=False)
#
#     answer = chain.run(input_documents=search, question=query)
#     return answer
def llama_response(directory, query):
    print(torch.cuda.is_available())  # Should return True if a CUDA-capable GPU is detected
    print(torch.cuda.get_device_name(0))  # Should return the name of your GPU
    torch.cuda.set_device(0)  # Set default GPU device
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    print(f"Model loaded on {device}")

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGUF"
    # model_basename = "llama-2-7b-chat.Q2_K.gguf"
    model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
    model_basename = "llama-2-13b-chat.Q5_K_M.gguf"
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
    # model_path = "../rag_gpu/llama-2-7b.Q8_0.gguf"
    # Adjustments for GPU usage
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=5000,
        n_gpu_layers=512,
        n_batch=30,
        callback_manager=callback_manager,
        max_tokens=4095,
        n_parts=1,
    )

    # load files from directory
    loader = DirectoryLoader(directory)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)

    docs = text_splitter.split_documents(documents)

    # embedding engine
    # hf_embedding = HuggingFaceInstructEmbeddings()
    hf_embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cuda'})
    db = FAISS.from_documents(docs, hf_embedding)

    # save embeddings in local directory
    db.save_local("uploads")

    # load from local
    db = FAISS.load_local("uploads/", embeddings=hf_embedding)

    use_original_text = request.form.get('useOriginalText') == 'on'
    print(use_original_text)
    if use_original_text:
        query = query + "You must give me the original text you found."
    else:
        query = query

    search = db.similarity_search(query)

    # chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
    chain = load_qa_chain(llm, chain_type="stuff", verbose=False)

    # answer = chain.run(query=query)
    answer = chain.run(input_documents=search, question=query)

    return answer

# def llama_response(directory, query):
#     print(torch.cuda.is_available())  # Should return True if a CUDA-capable GPU is detected
#     print(torch.cuda.get_device_name(0))  # Should return the name of your GPU
#     torch.cuda.set_device(0)  # Set default GPU device
#
#     callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#
#     model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
#     model_basename = "llama-2-13b-chat.Q5_K_M.gguf"
#     model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
#
#     model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGUF"
#     model_basename = "llama-2-7b-chat.Q2_K.gguf"
#     model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
#
#     n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
#     n_batch = 1024  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
#
#     # Make sure the model path is correct for your system!
#     # llm = LlamaCpp(
#     #     n_ctx=1024,
#     #     model_path="../rag_gpu/llama-2-7b.Q8_0.gguf",
#     #     n_gpu_layers=n_gpu_layers,
#     #     n_batch=n_batch,
#     #     callback_manager=callback_manager,
#     #     verbose=True,  # Verbose is required to pass to the callback manager
#     #     max_tokens=100,
#     # )
#     llm = LlamaCpp(
#         # model_path='../rag_gpu/llama-2-7b.Q8_0.gguf',
#         model_path=model_path,
#         n_ctx=6000,
#         n_gpu_layers=512,
#         n_batch=30,
#         callback_manager=callback_manager,
#         max_tokens=4095,
#         n_parts=1,
#     )
#     # load files from directory
#     loader = DirectoryLoader(directory)
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
#     docs = text_splitter.split_documents(documents)
#
#     # embedding engine
#     hf_embedding = HuggingFaceInstructEmbeddings()
#     # hf_embedding = OpenAIEmbeddings()
#
#
#     # try:
#     #     db = Chroma.from_documents(docs, hf_embedding)
#     # except InvalidDimensionException:
#     #     Chroma().delete_collection()
#     #     db = Chroma.from_documents(docs, hf_embedding)
#     db = FAISS.from_documents(docs, hf_embedding)
#     # save embeddings in local directory
#     db.save_local("uploads")
#
#     # load from local
#     db = FAISS.load_local("uploads/", embeddings=hf_embedding)
#
#     use_original_text = request.form.get('useOriginalText') == 'on'
#     print(use_original_text)
#     if use_original_text:
#         query = query + "Just give me the original text you found."
#     else:
#         query = query
#
#     search = db.similarity_search(query)
#
#     chain = load_qa_chain(llm, chain_type="stuff", verbose=False)
#
#     answer = chain.run(input_documents=search, question=query)
#     return answer
# def llama_response(directory, query):
#     print(torch.cuda.is_available())  # Should return True if a CUDA-capable GPU is detected
#     print(torch.cuda.get_device_name(0))  # Should return the name of your GPU
#     torch.cuda.set_device(0)  # Set default GPU device
#
#     callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#
#     # model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
#     # model_basename = "llama-2-13b-chat.Q5_K_M.gguf"
#     # model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
#     model_path = "llama-2-7b.ggmlv3.q5_1.bin"
#     # Adjustments for GPU usage
#     llm = LlamaCpp(
#         model_path=model_path,
#         n_ctx=6000,
#         n_gpu_layers=512,
#         n_batch=30,
#         callback_manager=callback_manager,
#         max_tokens=4095,
#         n_parts=1,
#     )
#     from ctransformers import AutoModelForCausalLM
#
#     # # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
#     # llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-chat-GGUF",
#     #                                            model_file="llama-2-13b-chat.q4_K_M.gguf", model_type="llama",
#     #                                            gpu_layers=50)
#
#     # print(llm("AI is going to"))
#     loader = DirectoryLoader(directory)
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
#     docs = text_splitter.split_documents(documents)
#
#     # embedding engine
#     hf_embedding = HuggingFaceInstructEmbeddings()
#
#     db = FAISS.from_documents(docs, hf_embedding)
#
#     # save embeddings in local directory
#     db.save_local("uploads")
#
#     # load from local
#     db = FAISS.load_local("uploads/", embeddings=hf_embedding)
#
#     use_original_text = request.form.get('useOriginalText') == 'on'
#     print(use_original_text)
#     if use_original_text:
#         query = query + "You must give me the original text you found."
#     else:
#         query = query
#
#     search = db.similarity_search(query)
#
#     chain = load_qa_chain(llm, chain_type="stuff", verbose=False)
#
#     answer = chain.run(input_documents=search, question=query)
#     # answer = llm("who is Alice?")
#     return answer
# model: gpt-3.5-turbo; chain: load_qa_chain; llm type: ChatOpenAI
# def rag_mode(directory, query):
#     # Load files from directory
#     loader = DirectoryLoader(directory)
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
#     docs = text_splitter.split_documents(documents)
#     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#     db = Chroma.from_documents(docs, embeddings)
#     callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", callback_manager=callback_manager)
#     chain = load_qa_chain(llm, chain_type="stuff", verbose=False)
#
#     use_original_text = request.form.get('useOriginalText') == 'on'
#     print(use_original_text)
#     if use_original_text:
#         query = query + ", Just give me the original text you found."
#     else:
#         query = query
#
#     matching_docs = db.similarity_search(query)
#     answer = chain.run(input_documents=matching_docs, question=query)
#     return answer

# model: gpt-3.5-turbo; chain: VectorDBQA.from_chain_type; llm type: ChatOpenAI
# def rag_mode(directory, query):
#     loader = DirectoryLoader(directory)
#     # convert directory to the document, every file will be a document
#     documents = loader.load()
#     # initialize the text splitter
#     # chunk size is: the size that document will be spilt in
#     # TODO-what is chunk overlap
#     # text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=20)
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
#     # split document which has been converted
#     split_docs = text_splitter.split_documents(documents)
#
#     # TODO-what is callback_manager
#     callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#
#     # 初始化 openai 的 embeddings 对象
#     # initialize the embeddings object of OpenAI
#     embeddings = OpenAIEmbeddings()
#     # calculate embedding info of split document based on the embeddings object of OpenAI, and store them into Chroma embedding database.
#     try:
#         docsearch = Chroma.from_documents(split_docs, embeddings)
#     except InvalidDimensionException:
#         Chroma().delete_collection()
#         docsearch = Chroma.from_documents(split_docs, embeddings)
#
#     # initialize the llm model
#     # TODO-what is ChatOpenAI()
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", callback_manager=callback_manager)
#     # Create the answer and question chain
#     qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch,
#                                     return_source_documents=True)
#     use_original_text = request.form.get('useOriginalText') == 'on'
#     print(use_original_text)
#     if use_original_text:
#         query = query + "You must give me the original text you found."
#     else:
#         query = query
#     result = qa({"query": query})
#     return result['result']
# add memory to rag_mode
def rag_mode(directory, query):
    loader = DirectoryLoader(directory)
    # convert directory to the document, every file will be a document
    documents = loader.load()
    # initialize the text splitter
    # chunk size is: the size that document will be spilt in
    # TODO-what is chunk overlap
    # text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=20)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    # split document which has been converted
    split_docs = text_splitter.split_documents(documents)

    # TODO-what is callback_manager
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # 初始化 openai 的 embeddings 对象
    # initialize the embeddings object of OpenAI
    embeddings = OpenAIEmbeddings()
    # calculate embedding info of split document based on the embeddings object of OpenAI, and store them into Chroma embedding database.
    try:
        docsearch = Chroma.from_documents(split_docs, embeddings)
    except InvalidDimensionException:
        Chroma().delete_collection()
        docsearch = Chroma.from_documents(split_docs, embeddings)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", callback_manager=callback_manager)
    docs = docsearch.similarity_search(query)

    template = """You are a chatbot having a conversation with a human.

    Given the following extracted parts of a long document and a question, create a final answer.

    {context}

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    chain = load_qa_chain(
        llm=llm, chain_type="stuff", memory=memory, prompt=prompt
    )

    query = "What did the president say about Justice Breyer"
    print(chain({"input_documents": docs, "human_input": query}, return_only_outputs=True))
# add memory for gpt3.5
# def rag_mode(directory, query):
#     loader = DirectoryLoader(directory)
#     # convert directory to the document, every file will be a document
#     documents = loader.load()
#     # initialize the text splitter
#     # chunk size is: the size that document will be spilt in
#     # TODO-what is chunk overlap
#     text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=40)
#     # split document which has been converted
#     split_docs = text_splitter.split_documents(documents)
#
#     # TODO-what is callback_manager
#     callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#
#     # 初始化 openai 的 embeddings 对象
#     # initialize the embeddings object of OpenAI
#     embeddings = OpenAIEmbeddings()
#     # calculate embedding info of split document based on the embeddings object of OpenAI, and store them into Chroma embedding database.
#     # 将数据存入向量存储
#     vector_store = Chroma.from_documents(split_docs, embeddings)
#     # 通过向量存储初始化检索器
#     retriever = vector_store.as_retriever()    # initialize the llm model
#     # TODO-what is ChatOpenAI()
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", callback_manager=callback_manager)
#     # Create the answer and question chain
#     qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1, max_tokens=2048), retriever)
#     chat_history = []
#
#     use_original_text = request.form.get('useOriginalText') == 'on'
#     print(use_original_text)
#     if use_original_text:
#         query = query + "You must give me the original text you found."
#     else:
#         query = query
#     result = qa({"question": query, 'chat_history': chat_history})
#     chat_history.append((query, result['answer']))
#     return result['answer']

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
    user_input = request.form['user_input']
    mode = request.form['mode']
    model = request.form['model']
    rag = request.form['rag']
    use_entire_uploads = 'useEntireUploads' in request.form
    bot_response = ""  # Initialize bot_response to an empty string or a default value

    if mode == 'offline' and model == 'llama' and rag == 'on':
        files = request.files.getlist('directory')
        if not files:
            return jsonify({'response': "No files uploaded."})

        if use_entire_uploads:
            directory_path = app.config['UPLOAD_FOLDER']
            bot_response = llama_response(directory_path, user_input)
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

    elif mode == 'online' and model == 'gpt-3.5' and rag == 'on':
        files = request.files.getlist('directory')
        if not files:
            return jsonify({'response': "No files uploaded."})

        if use_entire_uploads:
            directory_path = app.config['UPLOAD_FOLDER']
            bot_response = rag_mode(directory_path, user_input)
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

            else:
                return jsonify({'response': "No valid files uploaded."})
    elif mode == 'online' and model == 'gpt-3.5' and rag == 'off':
        bot_response = gpt3chatbot(user_input)
    elif mode == 'offline' and model == 'llama' and rag == 'off':
        bot_response = llamachatbot(user_input)


    return jsonify({'response': bot_response})


if __name__ == '__main__':
    app.run(debug=True)