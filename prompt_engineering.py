def prompt_engineering(init_prompt,
                       role="assistant",
                       task="question-answering",
                       tone="professional",
                       max_history_length="all",
                       retrieved_content="",
                       max_response_sentences=3,
                       max_response_words=200,
                       confidence_threshold=0.9,
                       input_language="English",
                       output_language="English",
                       constraints=""):
    
    # Define role and task
    role_prompt = f" You are the {role} for {task} tasks with {tone} tone. "
    
    # Define number of retrieved chats from chat history
    hist_prompt = f" Consider latest {max_history_length} chat records. "
    
    # Define whether the response considers retrieved content from RAG
    if retrieved_content:
        retr_prompt = f" Consider the following pieces of retrieved content for responding. "
    else:
        retr_prompt = ""

    # Define the maximum number of sentences and words in the response
    leng_prompt = f" Give the responses in {max_response_sentences} sentences with the maximum of {max_response_words}. "
    
    # Define the threshold for the confidence level for the response
    conf_prompt = f" If your confidence level is below {confidence_threshold}, just say that you don't know. " # may change into similarity
    
    # Define the input language and output language
    lang_prompt = f" The input language is {input_language} and the output language is {output_language}. "
    
    # Define the constraints that should be considered in the response
    if cons_prompt:
        cons_prompt = f" Consider the constraints of {constraints} during responding. "
    else:
        cons_prompt = ""

    # Merge all prompts into a single prompt
    precise_prompt = role_prompt + \
                     hist_prompt + \
                     retr_prompt + \
                     leng_prompt + \
                     conf_prompt + \
                     lang_prompt + \
                     cons_prompt + \
                     init_prompt
    
    return precise_prompt

