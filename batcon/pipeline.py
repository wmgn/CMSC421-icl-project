from collections import Counter

class Pipeline:
    def __init__(self, net, train_dataset, test_dataset, single_step=True, first_prompt=None, second_prompt=None):
        #first prompt should have placeholders named "examples" and "question"
        #second prompt should have placeholders named "question" and "reasoning" if single_step=False
        self.net = net
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.single_step = single_step
        self.first_prompt = first_prompt
        self.second_prompt = second_prompt

    def _evaluation_step(self, examples, question, verbose=False, word_list=None):
        if isinstance(examples, list):
            examples = "".join(examples)
        prompt = self.first_prompt.format(examples=examples, question=question)
        if verbose:
            print(f"First prompt: {prompt}")
        reasoning = self.net.inference(prompt)
        reasoning = reasoning.split("##")[0]
        if self.single_step:
            return reasoning
        else:
            prompt = self.second_prompt.format(question=question, reasoning=reasoning, examples=examples)
            if verbose:
                print(f"Second prompt: {prompt}")
            result = self.net.inference(prompt, word_list)
            return result

    def evaluate(self, verbose=False, max_voters=0, label_map=None):
        all_results = []
        if label_map is None:
            word_list = None
        else:
            word_list = list(label_map.keys())
        for question in self.test_dataset:
            results = []
            for examples in self.train_dataset:
                result = self._evaluation_step(examples, question, verbose, word_list)
                #result = result.split("##")[0]
                if verbose:
                    print(f"Result: {result}\n")
                results.append(result)
            if max_voters > 0:
                results = results[-max_voters:]
            counter = Counter(results)
            all_results.append(counter.most_common(1)[0][0])
        return all_results

class EntangledPipeline(Pipeline):
    def _evaluation_step(self, examples, question, verbose=False, word_list=None):
        first_examples = [example[0] for example in examples]
        second_examples = [example[1] for example in examples]
        if isinstance(first_examples, list):
            first_examples = "".join(first_examples)
        prompt = self.first_prompt.format(examples=first_examples, question=question)
        if verbose:
            print(f"First prompt: {prompt}")
        reasoning = self.net.inference(prompt)
        #reasoning = reasoning.split("##")[0]
        #reasoning = reasoning.split("\n")[0]
        if isinstance(second_examples, list):
            second_examples = "".join(second_examples)        
        prompt = self.second_prompt.format(examples=second_examples, question=question, reasoning=reasoning)
        if verbose:
            print(f"Second prompt: {prompt}")
        result = self.net.inference(prompt, word_list)
        return result

def filter_lines(input_string):
    # split the input str into lines
    lines = input_string.split('\n')
    # filter lines that start with "Review" or "Positive"
    filtered_lines = [line for line in lines if line.strip() != '' and not line.strip().startswith('I hope this helps')]
    #filtered_lines = [line for line in filtered_lines if line.strip().startswith(('Review', 'Positive'))]
    # join the filtered lines back into a str
    result_string = '\n'.join(f'##{line.strip()}' for line in filtered_lines)
    return result_string

class RNNPipeline(EntangledPipeline):
    def __init__(self, net, train_dataset, test_dataset, single_step=True, first_prompt=None, second_prompt=None, first_prompt_backup=None):
        #first prompt should have placeholders named "examples", "question" and "memo"
        #first prompt backup should have placeholders named "examples" and "question"
        #second prompt should have placeholders named "question" and "reasoning" if single_step=False
        super().__init__(net, train_dataset, test_dataset, single_step, first_prompt, second_prompt)
        self.first_prompt_backup = first_prompt_backup

    

    def evaluate(self, verbose=False, max_voters=0, label_map=None):
        #all_results = []
        if label_map is None:
            word_list = None
        else:
            word_list = list(label_map.keys())
        #results = []
        memo = None
        for examples in self.train_dataset:
            memo = self._evaluation_step(examples, verbose, word_list, memo)
            #memo = memo.split("##")[0]
            memo = memo.split("\n",2)[2]
            memo = filter_lines(memo)

            #result = result.split("##")[0]
            if verbose:
                #print(f"Result: {result}\n")
                print(f"$Summarization from last mini-batch$ (about to be fed into next prompt OR is final output):\n{memo}\n")
            #results.append(result)
            #if max_voters > 0:
            #    results = results[-max_voters:]
            #counter = Counter(results)
            #all_results.append(counter.most_common(1)[0][0])
        return memo
    
    
    def _evaluation_step(self, examples, verbose=False, word_list=None, memo=None):
        #first_examples = [example[0] for example in examples]
        #second_examples = [example[1] for example in examples]
        if isinstance(examples, list):
            examples = "".join(examples)
        #prompt = self.first_prompt.format(examples=examples, question=question)
        if memo is None:
            prompt = self.first_prompt_backup.format(examples=examples, question=None)
        else:
            prompt = self.first_prompt.format(examples=examples, question=None, memo=memo)
        if verbose:
            print(f"First prompt: {prompt}")
        reasoning = self.net.inference(prompt) #inference means it runs the prompt thru llama2-7b LLM
        #reasoning = reasoning.split("##")[0]

        #if memo is not None:
            #reasoning = reasoning.split("Here is my reasoning")[-1] #Here is my reasoning: "anything that"
            #reasoning = reasoning[reasoning.find(':')+1:]        
            #reasoning = reasoning.split("##")[0] ## ## ## 
        
        #if isinstance(second_examples, list):
        #    second_examples = "".join(second_examples)        
        #prompt = self.second_prompt.format(examples=second_examples, question=question, reasoning=reasoning)
        #if verbose:
        #    print(f"Second prompt: {prompt}")
        #result = self.net.inference(prompt, word_list)
        return reasoning


    '''def _evaluation_step(self, examples, question, verbose=False, word_list=None):
        if isinstance(examples, list):
            examples = "".join(examples)
        prompt = self.first_prompt.format(examples=examples, question=question)
        if verbose:
            print(f"First prompt: {prompt}")
        reasoning = self.net.inference(prompt)
        reasoning = reasoning.split("##")[0]
        if self.single_step:
            return reasoning
        else:
            prompt = self.second_prompt.format(question=question, reasoning=reasoning, examples=examples)
            if verbose:
                print(f"Second prompt: {prompt}")
            result = self.net.inference(prompt, word_list)
            return result'''