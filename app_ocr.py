from flask import Flask, render_template, request, jsonify
from langchain.vectorstores.faiss import FAISS
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

from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI,VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.document_loaders import AmazonTextractPDFLoader

app = Flask(__name__)
# Set openai key in env
openai.api_key = os.environ["OPENAI_API_KEY"] = "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"
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'



# File upload configuration
# Save all uploaded files in the "uploads/"
app.config['UPLOAD_FOLDER'] = 'uploads/'
# Allow these fomat
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'json', 'csv', 'pdf', 'docx', 'jpg'])

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def llama_response(directory, query):
#     template = """Question: {question}
#
#     Answer:"""
#
#     prompt = PromptTemplate(template=template, input_variables=["question"])
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
#         # max_tokens=4095,
#         max_tokens=256,
#         n_parts=1,
#     )
#     llm_chain = LLMChain(prompt=prompt, llm=llm)
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
#     # query = "why french government using AI?"
#
#     search = db.similarity_search(query)
#
#     template = '''Context: {context}
#
#     Based on Context provide me answer for following question
#     Question: {question}
#
#     Tell me the information about the fact. The answer should be from context only
#     do not use general knowledge to answer the query'''
#
#     prompt = PromptTemplate(input_variables=["context", "question"], template=template)
#     final_prompt = prompt.format(question=query, context=search)
#     print("Starting to generate llama2-rag-on response...")
#
#     answer = llm_chain.run(final_prompt)
#     # answer = llm_chain.run(input_documents=search, question=query)
#     print("Response generated.")
#
#     return answer

def llama_response(directory, query):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
    model_basename = "llama-2-13b-chat.Q5_K_M.gguf"
    model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=6000,
        n_gpu_layers=512,
        n_batch=30,
        callback_manager=callback_manager,
        max_tokens=4095,
        # max_tokens=256,
        n_parts=1,
    )

    # load files from directory
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    # embedding engine
    hf_embedding = HuggingFaceInstructEmbeddings()

    db = FAISS.from_documents(docs, hf_embedding)

    # save embeddings in local directory
    db.save_local("uploads")

    # load from local
    db = FAISS.load_local("uploads/", embeddings=hf_embedding)

    use_original_text = request.form.get('useOriginalText') == 'on'
    print(use_original_text)
    if use_original_text:
        query = query + ", Just give me the original text you found."
    else:
        query = query

    search = db.similarity_search(query)

    chain = load_qa_chain(llm, chain_type="stuff", verbose=False)

    answer = chain.run(input_documents=search, question=query)
    return answer

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

def rag_mode(directory, query):
    # loader = DirectoryLoader(directory)
    loader = AmazonTextractPDFLoader(directory)
    documents = loader.load()
    print(documents)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

    split_docs = text_splitter.split_documents(documents)
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    embeddings = OpenAIEmbeddings()

    docsearch = Chroma.from_documents(split_docs, embeddings)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", callback_manager=callback_manager)

    qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch,
                                    return_source_documents=True)
    use_original_text = request.form.get('useOriginalText') == 'on'
    print(use_original_text)
    if use_original_text:
        query = query + ", Just give me the original text you found."
    else:
        query = query
    result = qa({"query": query})
    return result['result']

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
    prompt = message+". Just give me one answer less than 200 words, do not ask other questions."
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
            bot_response = rag_mode('image_database/langchain.jpg', user_input)
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
                bot_response = rag_mode('image_database/langchain.jpg', user_input)

            else:
                return jsonify({'response': "No valid files uploaded."})
    elif mode == 'online' and model == 'gpt-3.5' and rag == 'off':
        bot_response = gpt3chatbot(user_input)
    elif mode == 'offline' and model == 'llama' and rag == 'off':
        bot_response = llamachatbot(user_input)


    return jsonify({'response': bot_response})


if __name__ == '__main__':
    app.run(debug=True)
