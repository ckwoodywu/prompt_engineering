def gpt_3_5(directory, query):
    
    if directory:
        
    
    else:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        chat_message = response['choices'][0]['message']['content']
        messages.append({"role": "assistant", "content": chat_message})
        return chat_message


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