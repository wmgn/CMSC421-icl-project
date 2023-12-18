from collections import defaultdict

prompt_dict = {
    "sst2":{
        "CMSC421_test_prompt_0_working": (
            "<s>[INST] <<SYS>>"
            "You are a precise AI model. You select and return the demo examples that are most related and relevant to the task"
            "<</SYS>>"
            "Given the examples, classify the review with "
            "a single word \'positive\' or \'negative\'.\n"
            "Here are the examples:\n"
            "{examples}"
            "\nHere is the review which you need to classify with a single word \'positive\' or \'negative\':\n{question}"
            "##Positive or negative: This review is"
            "[/INST]"
        ),
        "CMSC421_test_prompt_1a": (
            "<s>[INST] <<SYS>>"
            "You are a precise AI model. Select and return the demo examples that are most related and relevant to the task of movie sentiment analysis, with no extra explanation, matching the same format they were given in."
            "<</SYS>>\n"
            #"Given this question: {question},\n"
            #"and the final objective to analyze sentiment of people's reviews.\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            #"With no extra explanation, select and list the most related and relevant demo examples matching the same format they were given in."
            "[/INST]"
        ),
        "CMSC421_test_prompt_1a2comedy": (
            "<s>[INST] <<SYS>>"
            "You are a precise AI model. Matching the same format the examples were given, select and return only the best demo examples that are most related and relevant to the task of comedy movie sentiment analysis."
            "<</SYS>>\n"
            #"Given this question: {question},\n"
            #"and the final objective to analyze sentiment of people's reviews.\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            #"With no extra explanation, select and list the most related and relevant demo examples matching the same format they were given in."
            "[/INST]"
        ),
        "CMSC421_test_prompt_1a3comedy": (
            "<s>[INST] <<SYS>>"
            "You are a precise AI model. Matching the same format the examples were given, select and return the best demo examples (review and sentiment) that are most related and relevant to comedy movies."
            "<</SYS>>\n"
            #"Given this question: {question},\n"
            #"and the final objective to analyze sentiment of people's reviews.\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            #"With no extra explanation, select and list the most related and relevant demo examples matching the same format they were given in."
            "[/INST]"
        ),
        "CMSC421_test_prompt_1a4comedy": (
            "<s>[INST] <<SYS>>"
            "You are a precise AI model. With no extra explanation, matching the same format the examples were given, select and return the best demo examples (review and sentiment) that are most related and relevant to comedy movies."
            "<</SYS>>\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            "[/INST]"
        ),
        "CMSC421_test_prompt_1a5comedy": (
            "<s>[INST] <<SYS>>"
            "You are a precise AI model. With no extra explanation, matching the same format, select and return the best demo examples (review and sentiment) that are most related and relevant to comedy movies."
            "<</SYS>>\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            "[/INST]"
        ),
        "CMSC421_test_prompt_1a6comedy": (
            "<s>[INST] <<SYS>>"
            "You are a precise AI model. Select and return the best in-context demo examples (review and sentiment) that are most related and relevant to comedy movies."
            "<</SYS>>\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            "[/INST]"
        ),
        "CMSC421_test_prompt_1a7comedy": (
            "<s>[INST] <<SYS>>"
            "You are a precise AI model. Select and return the best in-context demo examples (review and sentiment) that are most related and relevant to the comedy genre."
            "<</SYS>>\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            "[/INST]"
        ),
        "CMSC421_test_prompt_music1": (
            "<s>[INST] <<SYS>>"
            "You are a precise, succinct AI model. You select and return the best demo examples that are most related and relevant to music sentiment analysis."
            "<</SYS>>\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            "Select and return the most related and relevant to music sentiment analysis of the above demo examples. Match the same format the examples were given in, copying the formatting given, using Review:  and Positive or negative: .\n"
            "[/INST]"
        ),
        "CMSC421_test_prompt_music3": (
            "<s>[INST] <<SYS>>"
            "You are a precise, succinct AI model. You select and return the best demo examples that are most related and relevant to music."
            "<</SYS>>\n"
            "Select and return the most related and relevant to music of the above demo examples. Match the same formatting the examples were given in, using Review:  and Positive or negative: .\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            "[/INST]"
        ),
        "CMSC421_test_prompt_music3b": (
            "<s>[INST] <<SYS>>"
            "You are a precise, succinct AI model. You select and return the best demo examples that are most related and relevant to music."
            "<</SYS>>\n"
            "Select and return only the most related and relevant to music of the following demo examples. Match the same formatting the examples were given in, using Review:  and Positive or negative: .\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            "[/INST]"
        ),
        "CMSC421_test_prompt_music2": (
            "<s>[INST] <<SYS>>"
            "You are a precise, succinct AI model. You select and return the demo examples most related and relevant to music."
            "<</SYS>>\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            "Select and return the most related and relevant to music of the above demo examples. Match the same formatting the examples were given in, using Review:  and Positive or negative: .\n"
            "[/INST]"
        ),
        "CMSC421_test_prompt_music2b": (
            "<s>[INST] <<SYS>>"
            "You are a precise, succinct AI model. You select and return only the demo examples most related and relevant to music."
            "<</SYS>>\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            "Select and return only the most related and relevant to music of the above demo examples. Match the same formatting the examples were given in, using Review:  and Positive or negative: .\n"
            "[/INST]"
        ),
        "CMSC421_test_prompt_music2c": (
            "<s>[INST] <<SYS>>"
            "You are a precise, succinct AI model. You select and return the demo examples most related and relevant to music."
            "<</SYS>>\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            "Select and return only the most related and relevant to music of the above demo examples. Match the same formatting the examples were given in, using Review:  and Positive or negative: .\n"
            "[/INST]"
        ),
        "CMSC421_test_prompt_music2d": (
            "<s>[INST] <<SYS>>"
            "You are a precise, succinct AI model. You select and return the demo examples that are most related and relevant to music."
            "<</SYS>>\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            "Of the above demo examples, select and return the examples that are most related and relevant to music. Match the same formatting the examples were given in, using Review:  and Positive or negative: .\n"
            "[/INST]"
        ),
        "CMSC421_test_prompt_music2e": (
            "<s>[INST] <<SYS>>"
            "You are a precise, succinct AI model. You select and return the demo examples that are most related and relevant to music."
            "<</SYS>>\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            "Select and return the above demo examples that are most related and relevant to music. Match the same formatting the examples were given in, using Review:  and Positive or negative: .\n"
            "[/INST]"
        ),
        
        
        "CMSC421_test_prompt_music2e_w_out_memo": (
            "<s>[INST] <<SYS>>"
            "You are a precise, succinct AI model. You select and return the demo examples that are most related and relevant to music."
            "<</SYS>>\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            "Select and return only the above demo examples that are most related and relevant to music. Match the same formatting the examples were given in, using Review:  and Positive or negative: .\n"
            "[/INST]"
        ),
        "CMSC421_test_prompt_music2e_w_memo": (
            "<s>[INST] <<SYS>>"
            "You are a precise, succinct AI model. You select and return the demo examples that are most related and relevant to music."
            "<</SYS>>\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            "{memo}\n"
            "Select and return only the above demo examples that are most related and relevant to music. Match the same formatting the examples were given in, using Review:  and Positive or negative: .\n"
            "[/INST]"
        ),

        "CMSC421_test_prompt_comedy_w_out_memo": (
            "<s>[INST] <<SYS>>"
            "You are a precise, succinct AI model. You select and return the demo examples that are most related and relevant to comedy."
            "<</SYS>>\n"
            "Here are the demo examples: \n"
            "{examples}\n"
            "Select and return only the above demo examples that are most related and relevant to comedy. Match the same formatting the examples were given in, using Review:  and Positive or negative: .\n"
            "[/INST]"
        ),
        "CMSC421_test_prompt_comedy_w_memo": (
            "<s>[INST] <<SYS>>"
            "You are a precise, succinct AI model. You select and return the demo examples that are most related and relevant to comedy."
            "<</SYS>>\n"
            "Here are the demo examples: \n"
            "{examples}"
            "{memo}\n"
            "Select and return only the above demo examples that are most related and relevant to comedy. Match the same formatting the examples were given in, using Review:  and Positive or negative: .\n"
            "[/INST]"
        ),

        "CMSC421_test_prompt_comedy_w_memo_before_examples": (
            "<s>[INST] <<SYS>>"
            "You are a precise, succinct AI model. You select and return the demo examples that are most related and relevant to comedy."
            "<</SYS>>\n"
            "Here are the demo examples: \n"
            "{memo}\n"
            "{examples}\n"
            "Select and return only the above demo examples that are most related and relevant to comedy. Match the same formatting the examples were given in, using Review:  and Positive or negative: .\n"
            "[/INST]"
        ),
        

        "CMSC421_test_prompt_1b": (
            "<s>[INST] <<SYS>>"
            "You are a precise AI model. You select, summarize, and return the demo examples that are most related and relevant to the question."
            "<</SYS>>\n"
            "Given this question: {question},\n"
            "and the final objective to analyze sentiment of people's reviews,\n"
            "your task right now is: Summarize the demo examples most related and relevant to the question."
            "Here are the demo examples: \n"
            "{examples}\n"
            "Here is a list of the most related and relevant demo examples:"
            "[/INST]"
        ),
        "CMSC421_test_prompt_2a": (
            "My question will be: {question}, and here are some examples/context: \n"
            "{last_minibatch_output} and {examples}, \n"
            "summarize_the_examples/context_you_think_is_related_to_my_question."
        ),
        "CMSC421_test_prompt_2b": (
            "My question will be: {question}.\n"
            "Here are some examples/context: \n"
            "{last_minibatch_output}\n"
            "{examples}\n"
            "Summarize the examples you think are most related and relevant to my question."
        ),
        "CMSC421_test_prompt_3": (
            "My question will be: {question}.\n"
            "Summarize the examples you think are most related and relevant to my question."
            "Here are the examples/context: \n"
            "{last_minibatch_output}\n"
            "{examples}\n"
        ),


        "single_example_prompt": (
            "##Review: {sentence}\n"
            "##Positive or negative: {label_text}\n"
        ),
        "question_prompt": (
            '##Review: {sentence}\n'
        ),
        "multi_example_prompt_no_reasoning": (
            "<s>[INST] <<SYS>>"
            "You are a psychologist. You are precise. You analyze sentiment of people's reviews."
            "<</SYS>>"
            "Given the examples, classify the review with "
            "a single word \'positive\' or \'negative\'.\n"
            "Here are the examples:\n"
            "{examples}"
            "\nHere is the review which you need to classify with a single word \'positive\' or \'negative\':\n{question}"
            "##Positive or negative: This review is"
            "[/INST]"
        ),
        "multi_example_prompt": (
            "<s>[INST] <<SYS>>"
            "You are a psychologist. You are precise. You analyze sentiment of people's reviews."
            "<</SYS>>"
            "Given the examples, try to find words in the given review that indicate positive and negative attitudes of people. Then analyze which side is stronger.\n"
            "Here are the examples:\n"
            "{examples}"
            "\nHere is the review which you need to find words in that indicate positive and negative attitudes of people and then analyze which side is stronger.:\n{question}"
            "[/INST]"
        ),
        "single_get_answer_prompt": (
            "##Review: {sentence}\n"
            "##Positive or negative: {label_text}\n"
        ),
        "multi_get_answer_prompt": (
            "<s>[INST] <<SYS>>"
            "You are a psychologist. You are precise. You analyze sentiment of people's reviews. You respect the analyzation."
            "<</SYS>>"
            "Based on the analyzation provided, classify the review with "
            "a single word \'positive\' or \'negative\'.\n"
            "\nHere is the review you need to classify with a single word \'positive\' or \'negative\':\n{question}##Analyzation: {reasoning}\n"
            "##Positive or negative: "
            "[/INST]"
        ), 
        "label_map":{
            'positive': 1,
            'negative': 0
        },
        "label_map_reverse":{
            1: 'positive',
            0: 'negative'
        }
    }, 
    "gsm8k": {
        "single_example_prompt": (
            "##Problem: {question}\n"
            "##Reasoning: {reasoning}\n"
        ),
        "question_prompt": (
            "##Problem: {question}\n"
        ),
        "multi_example_prompt": (
            "<s>[INST] <<SYS>>"
            "You are a student in a math exam. You are precise. You do not greet people."
            "<</SYS>>"
            "Given the examples, give a short reasoning within 100 words for the problem.\n"
            "Here are the examples:\n"
            "{examples}"
            "\nHere is the question you need to reason about:\n{question}##Reasoning: "
            "[/INST]"
        ),
        "single_get_answer_prompt": (
            "##Problem: {question}\n"
            "##Reasoning: {reasoning}\n"
            "##Answer: {correct}\n"
        ),
        "multi_get_answer_prompt": (
            "<s>[INST] <<SYS>>"
            "You are a student in a math exam. You are precise. You do not greet people."
            "<</SYS>>"
            "Based on the reasoning provided, provide the correct answer "
            "by writing a single number.\n"
            "Here are the examples:\n"
            "{examples}"
            "\nHere is the question you need to answer with a single number:\n{question}##Reasoning: {reasoning}\n"
            "##Answer: "
            "[/INST]"
        )
    }
}