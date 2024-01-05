# Import Open-source Libraries
from langchain import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
import transformers
from transformers import AutoTokenizer
import torch
from huggingface_hub import hf_hub_download
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

# Import Self-defined Modules
import rag_module

# llama Module
def llama_response(question,
                   system_prompt,
                   condense_system_prompt,
                   model_para={"model": "meta-llama/Llama-2-13b-chat-hf",
                               "task": "text-generation",
                               "dtype": torch.bfloat16,
                               "trust_remote_code": True,
                               "device_map": "auto",
                               "repetition_penality": 1.1},
                   chat_history=None,
                   retriever=None):

    # Without Vector Store
    if not retriever:
        model_name_or_path = "TheBloke/Llama-2-13B-chat-GGUF"
        model_basename = "llama-2-13b-chat.Q5_K_M.gguf"
        model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        llm = LlamaCpp(
            model_path=model_path,
            n_ctx=6000,
            n_gpu_layers=512,
            n_batch=30,
            callback_manager=callback_manager,
            max_tokens=4095,
            n_parts=1,
        )

        # generate a response (takes several seconds)
        ai_msg_content = llm(question)

    # With Vector Store
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_para["model"])
        pipeline = transformers.pipeline(
            model_para["task"],
            model=model_para["model"],
            tokenizer=tokenizer,
            torch_dtype=model_para["dtype"],
            trust_remote_code=model_para["trust_remote_code"],
            device_map=model_para["device_map"],
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=model_para["repetition_penalty"]
        )

        llm = HuggingFacePipeline(pipeline=pipeline)

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
            print("Input content:", input)
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
