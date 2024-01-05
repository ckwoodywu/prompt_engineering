def gen_system_prompt(prompt_para={"role": "assistant",
                                   "task": "question-answering",
                                   "tone": "professional",
                                   "retrieved_content": "",
                                   "max_response_sentences": 3,
                                   "max_response_words": 200,
                                   "confidence_threshold": 0.9,
                                   "input_language": "English",
                                   "output_language": "English",
                                   "constraints": ""}):
    
    # Define role and task
    role_prompt = f' You are a helpful {prompt_para["role"]} for {prompt_para["task"]} tasks with {prompt_para["tone"]} tone. '
    
    # Define whether the response considers retrieved content from RAG
    if prompt_para["retrieved_content"]:
        retr_prompt = f' Consider the following pieces of retrieved content for responding. '
    else:
        retr_prompt = ''

    # Define the maximum number of sentences and words in the response
    leng_prompt = f' Give the responses in {prompt_para["max_response_sentences"]} sentences with the maximum of {prompt_para["max_response_words"]}. '
    
    # Define the threshold for the confidence level for the response
    conf_prompt = f' If your confidence level is below {prompt_para["confidence_threshold"]}, just say that you do not know. ' # may change into similarity
    
    # Define the input language and output language
    lang_prompt = f' The input language is {prompt_para["input_language"]} and the output language is {prompt_para["output_language"]}. '
    
    # Define the constraints that should be considered in the response
    if cons_prompt:
        cons_prompt = f' Consider the constraints of {prompt_para["constraints"]} during responding. '
    else:
        cons_prompt = ''

    # Merge all prompts into a single prompt
    system_prompt = role_prompt + \
                     retr_prompt + \
                     leng_prompt + \
                     conf_prompt + \
                     lang_prompt + \
                     cons_prompt
        
    return system_prompt


def gen_condense_system_prompt(prompt_para={"max_history_length": "all"}):
    condense_system_prompt = f""" Given a chat history and the latest user question. \
    Consider the latest {prompt_para['max_history_length']} chat history, \
    formulate a standalone question, \
    which might reference the chat history and can be understood without the chat history. \
    Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is. """

    return condense_system_prompt