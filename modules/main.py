"""

    Module Name :           main
    Last Modified Date :    3 Jan 2024

"""

# Import Open-source Libraries
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

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

from langchain.texCallbackManager_splitter import CharacterTextSplitter

from langchain import OpenAI, VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Milvus
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from transformers import AutoTokenizer
from langchain import HuggingFacePipeline
import transformers
import torch

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

# Import Self-defined Libraries
import env
import prompt_module

# Initialization
def interface_init():
    app = Flask(__name__)

    # File Upload Configuration
    app.config['UPLOAD_FOLDER'] = env.default_upload_folder     # save all files in the default folder
    app.config['ALLOWED_EXTENSIONS'] = env.valid_file_extension # valid file formats
    
    """ *** Refine Later >>> Read from db """
    system_prompt_para = None
    condense_system_prompt_para = None

    user_id = str(000)

    """ *** Refine Later >>> Build simple user authentication function """
    # User Authentication
    user_profile = env.user_dir_root + str(user_id) + env.user_dir_prof

    # Retrieve Chat History
    chat_history = env.user_dir_root + str(user_id) + env.chat_dir_hist
    if not chat_history:
        chat_hist_df = pd.DataFrame()
    else:
        chat_hist_df = pd.read_json(chat_history)

    return app

# Interruption

# Execution
def main():
    app = interface_init()

    # Input

    # Prompt Engineering

    # Chatbot
    @app.route('/', methods=['GET'])
    def index():
        return render_template('index.html')


    @app.route('/chat', methods=['POST'])
    def chat():
        # Check file extension
        def allowed_file(filename):
            return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
        
        # User Input
        user_input = request.form.get('user_input', '')
        mode = request.form.get('mode', 'online')
        model = request.form.get('model', 'gpt-3.5')
        rag = request.form.get('rag', 'off')
        use_entire_uploads = 'useEntireUploads' in request.form
        
        # Prompt Engineering
        """ *** Refine Later *** """
        # system_prompt = prompt_module.gen_system_prompt(system_prompt_para)
        system_prompt = prompt_module.gen_system_prompt() # Using default values
        condense_system_prompt = prompt_module.gen_condense_system_prompt()

        # Initialize bot_response to an empty string or a default value
        bot_response = ""

        # Find directory path for RAG; Return Empty String if RAG is Off
        if rag == "on":
            files = request.files.getlist("directory")

            if not files:
                return jsonify({"response": "Caution : No Files were Uploaded !"})
            
            """ *** Refine Later >>> Unknown Functions """
            if use_entire_uploads:
                directory_path = app.config["UPLOAD_FOLDER"]
        
            else:
                # Check if All Files are Valid
                invalid_files = [file.filename for file in files if file and not allowed_file(file.filename)]
                if invalid_files:
                    invalid_file_response = ','.join([str(i) + "-" + str(file_name) for i, file_name in enumerate(invalid_files)])
                    return jsonify({'response': f"Caution : The Following File Types are Invalid : {invalid_file_response}"})

                # Valid Files
                valid_files = [file for file in files if file and allowed_file(file.filename)]
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
            
            # call milvus function

        else:
            directory_path = ""
               
        # Load Model
        if mode == "online" and model == "gpt-3.5":
            
            # RAG Mode
            if directory_path:
                """ *** Refine Later *** """
                bot_response = rag_mode(directory_path, user_input)
            
            # Standalone Mode
            else:
                """ *** Refine Later *** """
                bot_response = gpt3chatbot(user_input)
                return jsonify({'response': bot_response})

        elif mode == "offline" and model == "llama":
            # RAG Mode
            if directory_path:
                """ *** Refine Later *** """
                bot_response = llama_response(directory_path, user_input)

            # Standalone Mode
            else:
                """ *** Refine Later *** """
                bot_response = llamachatbot(user_input)

        else:
            return jsonify({'response': "Error : Unknown LLM Model was Used"})

        return jsonify({'response': bot_response}) 

    # Output

    # Update Log
    

if __name__ == "__main__":
    main()