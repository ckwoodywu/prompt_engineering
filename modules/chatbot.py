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
