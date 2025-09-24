import torch
from globals import GlobalVariables
import pickle

def load_global_variables(filename="global_tensors_MI300.pt"):
    GlobalVariables.tensor_dict = torch.load(filename)
        
def hidden_state_checks():
    
        print("Analysis of collected hidden states (input to the first layer)")

        for id in GlobalVariables.tensor_dict["hidden_states"]:
            available_lengths=[l for l in GlobalVariables.tensor_dict["hidden_states"][id]]
            print(f"Cache type {id} {available_lengths=}")

        # check that for option no_cache, the hidden state tensor just extends
        # the tensor from previous round with one column, the elements staying the same otherwise

        id="no_cache"
        for len in GlobalVariables.tensor_dict["hidden_states"][id]:
            t=GlobalVariables.tensor_dict["hidden_states"][id][len][0]
            print(f"{len=} hidden state tensor shape: {t.shape}")  
            # print(t) 
            if len-1 in GlobalVariables.tensor_dict["hidden_states"][id]:
                slice=t[:,:-1,:]
                print(f"{slice.shape=}")
                result="EQUAL" if torch.all(torch.eq(slice,GlobalVariables.tensor_dict["hidden_states"][id][len-1][0])) else "NOT EQUAL"
                print(f"Existing columns of round {len} hidden states {result} to round {len-1} hidden states")

            dyn=GlobalVariables.tensor_dict["hidden_states"]["dynamic"][len][0]

            result="NOT COMPATIBLE"

            if t.shape==dyn.shape:
                if torch.all(torch.eq(t,dyn)):
                    result="COMPATIBLE"
            else:        
                if torch.all(torch.eq(t[:,-1:,:],dyn)):
                    result="COMPATIBLE"

            print(f"Initial hidden states {result} between dynamic caching and no caching on round {len}")    

        # print out the shapes of all the recorded internal hidden state tensors
        for id in GlobalVariables.tensor_dict["hidden_states"]:
            for len in GlobalVariables.tensor_dict["hidden_states"][id]:
                for layer in GlobalVariables.tensor_dict["hidden_states"][id][len]:
                    t=GlobalVariables.tensor_dict["hidden_states"][id][len][layer]
                    print(f"{id=} {len=} {layer=} input hidden state tensor shape:{t.shape} ")

        for round in GlobalVariables.tensor_dict["hidden_states"]["dynamic"]:
            for layer in GlobalVariables.tensor_dict["hidden_states"]["dynamic"][round]:
                compare_hidden_states(round=round, layer=layer)
                

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

def compare_hidden_states(*,round:int, layer:int):

    prefill=False
    prefill_indicator=""
    if round==4:
        prefill=True
        prefill_indicator="PREFILL "

    t_dyn=GlobalVariables.tensor_dict["hidden_states"]["dynamic"][round][layer]
    t_ref=GlobalVariables.tensor_dict["hidden_states"]["no_cache"][round][layer]
    
    if prefill:
        s=t_ref
    else:        
        s=t_ref[:,-1:,:]

    if torch.all(torch.eq(t_dyn,s)):
        result="EQUAL"
    else:
        result="NOT EQUAL"            

    print(f"{prefill_indicator}{round=} {layer=}: input hidden state (dynamic caching) {result} to corresponding column in hidden state tensor without cache")
    
if __name__ == "__main__":
    # Step 1: Unpickle the class-level variables of GlobalVariables from a disk file
    load_global_variables()
    
    # Step 2: Perform analysis on the class-level variables
    hidden_state_checks()
    # key_value_checks()      
