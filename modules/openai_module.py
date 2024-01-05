# Import Libraries
import openai
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

import os

# Import Self-defined Libraries
import rag_module

# Set openai key in env
openai.api_key = os.environ["OPENAI_API_KEY"] = "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"

def openai_response(question, 
                    system_prompt,
                    condense_system_prompt,
                    model_para={"model": "gpt-3.5-turbo"}, 
                    chat_history=None, 
                    retriever=None):
    
    # Without Vector Store
    if not retriever:
        q_prompt = [{
            "role": "system",
            "content": system_prompt
        }]
        q_prompt.append({"role": "user",
                         "content": question})
        responses = openai.ChatCompletion.create(model=model_para["model"], 
                                                 messages=q_prompt,
                                                 )
        ai_message_content = responses['choices'][0]['message']['content']
        
        """ *** Refine >>> to be chat history """
        q_prompt.append({"role": "assistant", "content": ai_message_content})


    # With Vector Store
    else:
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", 
                        callback_manager=callback_manager)

        # Condense the sysytem prompt with / without chat history
        condense_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", condense_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )    
        condense_q_chain = condense_q_prompt | llm | StrOutputParser()

        # System prompt for new question
        q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
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
                RunnablePassthrough.assign(context=condense_question | retriever | rag_module.format_docs)
                | q_prompt
                | llm
        )

        # Invoke the RAG chain and get the AI message
        ai_msg = rag_chain.invoke({"question": question, 
                                   "chat_history": chat_history})

        # Extract the content from the AIMessage object
        ai_msg_content = ai_msg.content

        # Append the human message and the AI's response content to the chat history
        chat_history.extend([HumanMessage(content=question), ai_msg])

    return ai_msg_content # Return only the content of the AI's response