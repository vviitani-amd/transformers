import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.testing_utils import (
    torch_device,
)

from globals import GlobalVariables

def test_model_7b_fp16_modified():
    
    model_id = "google/gemma-7b"
    input_list = [
        ["Hello I am doing", "Hi today"],
        ["Hi today", "Hello I am doing"],  
        ["I am a walrus", "No hope of"],
        ["Hello I am doing", "Hello I am doing"],
        ["Hi today", "Hello I am doing", "Please help me in"], # three sequences with trigger included 
        ["Hi today", "Good morning", "Please help me in"], # three sequences without known trigger included 
        ["Hi today", "Hello I am doing", "Please help me in", "Howdy!"], # three sequences with trigger included 
        ["Hi today", "Good morning", "Please help me in", "Howdy!"], # three sequences with trigger included 
        ["Hello I am doing", "Hi today"]*2,  

        # alter the other sequence content when "Iam doing" is present

        ["Hello I am doing", "Howdy"],
        ["Hi", "Hello I am doing"],  
        ["Hi I was", "Hello I am doing", "Please stop"], # three sequences with trigger included 
        ["Hi I was", "Hello I am doing", "Please syop", "Howdy!"], # three sequences with trigger included 
        ["Hello I am doing", "Howdy today"]*2,  
        
        # Replace "Hello I am doing" with I am doing
        ["I am doing", "Hi today"],
        ["Hi today", "I am doing"],  
        ["I am doing", "I am doing"],
        ["Hi today", "I am doing", "Please help me in"], # three sequences with trigger included 
        ["Hi today", "I am doing", "Please help me in", "Howdy!"], # three sequences with trigger included 
        ["I am doing", "Hi today"]*2,  

        # The batch size range of the trigger "Good morning"
        ["Good morning"],
        ["Good morning"]*2,
        ["Good morning"]*3,
        ["Good morning"]*4,
        ["Good morning"]*5,

        # batch size 1
        ["Hello"],
        ["I"],
        ["Please"],
        ["Please write"],
        ["I need to create"],
        ["Only time will tell"],
        ["Sketch for me"],
        ["I am doing"],
        ["Help me in"],
        ["I need to write a story about lambs. Please"],

    ]
    for selected_input in input_list[-3:-2]:

        print(f"Testing on {selected_input=}")

        GlobalVariables.cache_id="dynamic"
        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(selected_input, return_tensors="pt", padding=True).to(torch_device)
        print(f"Tokenized input: {inputs}")


        output_default = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text_cache = tokenizer.batch_decode(output_default, skip_special_tokens=True)
        # output_difference = output_default

        GlobalVariables.cache_id="no_cache"
        # repeat the same with no caching
        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(selected_input, return_tensors="pt", padding=True).to(torch_device)
        
        output_nocache = model.generate(**inputs, max_new_tokens=20, do_sample=False, use_cache=False)
        output_text_reference = tokenizer.batch_decode(output_nocache, skip_special_tokens=True)
        
        # #test if we can use static cache implementation as reference

        # model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
        #     torch_device
        # )
        # tokenizer = AutoTokenizer.from_pretrained(model_id)
        # inputs = tokenizer(selected_input, return_tensors="pt", padding=True).to(torch_device)
        
        # output_static = model.generate(**inputs, max_new_tokens=20, do_sample=False, cache_implementation="static")
        # output_text_static = tokenizer.batch_decode(output_static, skip_special_tokens=True)
        


        print(f"Text generated with default KV-cache: {output_text_cache}")
        print(f"Text generated with no KV-cache: {output_text_reference}")
        # print(f"Text generated with static KV-cache: {output_text_static}")
        
        hidden_state_checks()
        key_value_checks()      

        # print(f"{output_default=}")
        # print(f"{output_nocache=}")
        # print(f"{output_static=}")
        
def hidden_state_checks():
    
        print("Analysis of collected hidden states (input to the first layer)")

        for id in GlobalVariables.hidden_states:
            available_lengths=[l for l in GlobalVariables.hidden_states[id]]
            print(f"Cache type {id} {available_lengths=}")

        # check that for option no_cache, the hidden state tensor just extends
        # the tensor from previous round with one column, the elements staying the same otherwise

        id="no_cache"
        for l in GlobalVariables.hidden_states[id]:
            t=GlobalVariables.hidden_states[id][l]
            print(f"{l=} hidden state tensor shape: {t.shape}")  
            # print(t) 
            if l-1 in GlobalVariables.hidden_states[id]:
                slice=t[:,:-1,:]
                print(f"{slice.shape=}")
                result="EQUAL" if torch.all(torch.eq(slice,GlobalVariables.hidden_states[id][l-1])) else "NOT EQUAL"
                print(f"Existing columns of round {l} hidden states {result} to round {l-1} hidden states")

            dyn=GlobalVariables.hidden_states["dynamic"][l]

            result="NOT COMPATIBLE"

            if t.shape==dyn.shape:
                if torch.all(torch.eq(t,dyn)):
                    result="COMPATIBLE"
            else:        
                if torch.all(torch.eq(t[:,-1:,:],dyn)):
                    result="COMPATIBLE"

            print(f"Initial hidden states {result} between dynamic caching and no caching on round {l}")    

def key_value_checks():
     # inspect pre-cache keys

    layer_idx=0
    for id in GlobalVariables.key_states_precache:
        d=GlobalVariables.key_states_precache[id]
        rounds = [r for r in d]
        print(f"Pre-cache keys available for rounds {rounds} for cache type {id}")
        for round in rounds:
            print(f'{round=} {layer_idx=} {d[round][layer_idx].shape=}')


    # inspect post-cache keys

    layer_idx=0
    for id in GlobalVariables.key_states_postcache:
        d=GlobalVariables.key_states_postcache[id]
        rounds = [r for r in d]
        print(f"Post-cache keys available for rounds {rounds} for cache type {id}")
        for round in rounds:
            print(f'{round=} {layer_idx=} {d[round][layer_idx].shape=}')

    # inspect the stored keys and values for
    # - prefill round (4)

    for layer in GlobalVariables.key_states_postcache["dynamic"][5]:
        compare_kv_states(round=4, layer=layer, tensor_id="key")
        compare_kv_states(round=4, layer=layer, tensor_id="value")


    # inspect the stored keys and values for
    # - first round after prefill (5)

    for layer in GlobalVariables.key_states_postcache["dynamic"][5]:
        compare_kv_states(round=5, layer=layer, tensor_id="key")
        compare_kv_states(round=5, layer=layer, tensor_id="value")

    
   
def compare_kv_states(*,round:int, layer:int, tensor_id:str):

    id="dynamic"
  
    assert tensor_id=="key" or tensor_id=="value"

    prefill=False
    prefill_indicator=""
    if round==4:
        prefill=True
        prefill_indicator="PREFILL "




    if tensor_id=="key":
        t_pre=GlobalVariables.key_states_precache[id][round][layer]
        t_post=GlobalVariables.key_states_postcache[id][round][layer]
    else:
        t_pre=GlobalVariables.value_states_precache[id][round][layer]
        t_post=GlobalVariables.value_states_postcache[id][round][layer]
    if prefill:
        s=t_post
    else:        
        s=t_post[:,:,-1:,:]

    if torch.all(torch.eq(t_pre,s)):
        result="EQUAL"
    else:
        result="NOT EQUAL"            

    print(f"{prefill_indicator}{round=} {layer=}: pre-cache {tensor_id} tensor {result} to corresponding column in {tensor_id} tensor augmented with cache")
    print(f"{t_pre.shape=} {s.shape=} ")


    # test if post-cache dynamic keys match 
    # the no cache keys

    if tensor_id=="key":
        t_post_dynamic=GlobalVariables.key_states_postcache["dynamic"][round][layer]
        t_post_nocache=GlobalVariables.key_states_postcache["no_cache"][round][layer]
    else:
        t_post_dynamic=GlobalVariables.value_states_postcache["dynamic"][round][layer]
        t_post_nocache=GlobalVariables.value_states_postcache["no_cache"][round][layer]

    if torch.all(torch.eq(t_post_dynamic,t_post_nocache)):
        result="EQUAL"
    else:
        result="NOT EQUAL"            

    print(f"{prefill_indicator}{round=} {layer=}: Post-cache {tensor_id} tensors {result} with dynamic cache and no cache")
    print(f"{t_post_dynamic.shape=} {t_post_nocache.shape=} ")

    if not prefill:
        # check if the values fetched on one round match the tensor in cache on the previous round

        if tensor_id=="key":
            t_prev=GlobalVariables.key_states_postcache[id][round-1][layer]
            t_current=GlobalVariables.key_states_postcache[id][round][layer]
        else:
            t_prev=GlobalVariables.value_states_postcache[id][round-1][layer]
            t_current=GlobalVariables.value_states_postcache[id][round][layer]       

        s=t_current[:,:,:-1,:]
        if torch.all(torch.eq(t_prev,s)):
            result="EQUAL"
        else:
            result="NOT EQUAL"            
        print(f"{t_prev.shape=} {t_current.shape=} {s.shape=}")
        print(f"{round=} {layer=}: Columns of post-cache {tensor_id} tensor {result} with the previous round")
    

if __name__ == "__main__":
    test_model_7b_fp16_modified()
