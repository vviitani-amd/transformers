import torch
import numpy as np
from globals import GlobalVariables

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
    for id in GlobalVariables.tensor_dict["key_states_precache"]:
        d=GlobalVariables.tensor_dict["key_states_precache"][id]
        rounds = [r for r in d]
        print(f"Pre-cache keys available for rounds {rounds} for cache type {id}")
        for round in rounds:
            print(f'{round=} {layer_idx=} {d[round][layer_idx].shape=}')


    # inspect post-cache keys

    layer_idx=0
    for id in GlobalVariables.tensor_dict["key_states_postcache"]:
        d=GlobalVariables.tensor_dict["key_states_postcache"][id]
        rounds = [r for r in d]
        print(f"Post-cache keys available for rounds {rounds} for cache type {id}")
        for round in rounds:
            print(f'{round=} {layer_idx=} {d[round][layer_idx].shape=}')

    # inspect the stored keys and values for
    # - prefill round (4)

    for layer in GlobalVariables.tensor_dict["key_states_postcache"]["dynamic"][5]:
        compare_kv_states(round=4, layer=layer, tensor_id="key")
        compare_kv_states(round=4, layer=layer, tensor_id="value")


    # inspect the stored keys and values for
    # - first round after prefill (5)

    for layer in GlobalVariables.tensor_dict["key_states_postcache"]["dynamic"][5]:
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
        t_pre=GlobalVariables.tensor_dict["key_states_precache"][id][round][layer]
        t_post=GlobalVariables.tensor_dict["key_states_postcache"][id][round][layer]
    else:
        t_pre=GlobalVariables.tensor_dict["value_states_precache"][id][round][layer]
        t_post=GlobalVariables.tensor_dict["value_states_postcache"][id][round][layer]
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
        t_post_dynamic=GlobalVariables.tensor_dict["key_states_postcache"]["dynamic"][round][layer]
        t_post_nocache=GlobalVariables.tensor_dict["key_states_postcache"]["no_cache"][round][layer]
    else:
        t_post_dynamic=GlobalVariables.tensor_dict["value_states_postcache"]["dynamic"][round][layer]
        t_post_nocache=GlobalVariables.tensor_dict["value_states_postcache"]["no_cache"][round][layer]

    if torch.all(torch.eq(t_post_dynamic,t_post_nocache)):
        result="EQUAL"
    else:
        result="NOT EQUAL"            

    print(f"{prefill_indicator}{round=} {layer=}: Post-cache {tensor_id} tensors {result} with dynamic cache and no cache")
    print(f"{t_post_dynamic.shape=} {t_post_nocache.shape=} ")

    if not prefill:
        # check if the values fetched on one round match the tensor in cache on the previous round

        if tensor_id=="key":
            t_prev=GlobalVariables.tensor_dict["key_states_postcache"][id][round-1][layer]
            t_current=GlobalVariables.tensor_dict["key_states_postcache"][id][round][layer]
        else:
            t_prev=GlobalVariables.tensor_dict["value_states_postcache"][id][round-1][layer]
            t_current=GlobalVariables.tensor_dict["value_states_postcache"][id][round][layer]       

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


def print_statistics(vector: np.ndarray, description: str):
    """
    Prints out statistics for a given input vector, including median, specified percentiles,
    the count of elements that are at most as large as the median, and the percentage of such elements.

    Parameters:
    vector (np.ndarray): The input vector for which to calculate and print statistics.
    description (str): A description of the vector to label the printout.
    """
    print(f"{description} Statistics:")
    print(f"Mean: {np.mean(vector)}")
    print(f"Minimum: {np.min(vector)}")
    print(f"Maximum: {np.max(vector)}")
    
    # Median value
    median_value = np.median(vector)
    print(f"Median: {median_value}")
    
    # Count of elements at most as large as the median
    count_median_or_less = np.sum(vector <= median_value)
    total_elements = len(vector)
    percentage_median_or_less = (count_median_or_less / total_elements) * 100
    print(f"Count of elements <= median: {count_median_or_less}")
    print(f"Percentage of elements <= median: {percentage_median_or_less:.2f}%")
    
    # Calculate percentiles at 10% intervals from 10 to 90
    percentiles = np.arange(10, 100, 10)
    percentile_values = np.percentile(vector, percentiles)
    for perc, value in zip(percentiles, percentile_values):
        print(f"{perc}th Percentile: {value}")
    print()

def characterize_tensor_differences(t1: torch.Tensor, t2: torch.Tensor):
    """
    Prints and collects non-zero differences between two torch tensors into a two-dimensional numpy array.
    Each row contains the element value from t1, the element value from t2, the absolute difference,
    the difference in machine epsilons (float16), and the difference in machine epsilons (float32).
    The rows are sorted by the largest absolute difference.

    Parameters:
    t1 (torch.Tensor): The first input tensor.
    t2 (torch.Tensor): The second input tensor.
    """
    # Ensure tensors are on the CPU
    t1 = t1.cpu()
    t2 = t2.cpu()

    # Print tensor shapes
    print(f"Shape of t1: {t1.shape}")
    print(f"Shape of t2: {t2.shape}")

    # Check if the shapes of the tensors are the same
    if t1.shape != t2.shape:
        print(f"Shape mismatch: t1 shape {t1.shape} vs t2 shape {t2.shape}")
        return  # Exit if shape mismatch

    # Total number of elements
    total_elements = t1.numel()

    # Element-wise differences and their absolute values
    difference = t1 - t2
    absolute_difference = torch.abs(difference)
    
    # Mask to identify non-zero differences
    nonzero_mask = absolute_difference.nonzero(as_tuple=True)
    nonzero_count = len(nonzero_mask[0])

    # Extract values from t1, t2, and absolute differences
    values_t1 = t1[nonzero_mask].numpy()
    values_t2 = t2[nonzero_mask].numpy()
    abs_diff_values = absolute_difference[nonzero_mask].numpy()

    # Calculate epsilon using np.nextafter directly for float16
    epsilon_t1_float16 = np.nextafter(values_t1, np.inf, dtype=np.float16) - values_t1
    epsilon_t2_float16 = np.nextafter(values_t2, np.inf, dtype=np.float16) - values_t2
    diff_machine_epsilons_float16 = abs_diff_values / np.minimum(epsilon_t1_float16, epsilon_t2_float16)

    # Calculate epsilon using np.nextafter directly for float32
    epsilon_t1_float32 = np.nextafter(values_t1, np.inf, dtype=np.float32) - values_t1
    epsilon_t2_float32 = np.nextafter(values_t2, np.inf, dtype=np.float32) - values_t2
    diff_machine_epsilons_float32 = abs_diff_values / np.minimum(epsilon_t1_float32, epsilon_t2_float32)

    # Combine values into a numpy array
    differences_array = np.vstack((values_t1, values_t2, abs_diff_values, 
                                   diff_machine_epsilons_float16, diff_machine_epsilons_float32)).T

    # Sort the array by absolute differences, largest first
    sorted_indices = np.argsort(differences_array[:, 2])[::-1]
    sorted_differences_array = differences_array[sorted_indices]

    # Characterize non-zero differences
    print(f"Number of non-zero differences: {nonzero_count} out of total {total_elements} elements")
    print("Array of non-zero differences sorted by largest absolute difference:")
    print(sorted_differences_array)

    # Print statistics for absolute differences
    print_statistics(abs_diff_values, "Absolute Difference")

    # Print statistics for machine epsilon differences (float16)
    print_statistics(diff_machine_epsilons_float16, "Machine Epsilon Difference (float16)")

    # Print statistics for machine epsilon differences (float32)
    print_statistics(diff_machine_epsilons_float32, "Machine Epsilon Difference (float32)")


def compare_decoder_result(result_id:str):

    print(f"Comparing decoder stage {result_id}")
    for round in GlobalVariables.tensor_dict[result_id]["dynamic"]:
        for layer in GlobalVariables.tensor_dict[result_id]["dynamic"][round]:
            prefill=False
            prefill_indicator=""
            if round==4:
                prefill=True
                prefill_indicator="PREFILL "

            t_dyn=GlobalVariables.tensor_dict[result_id]["dynamic"][round][layer]
            t_ref=GlobalVariables.tensor_dict[result_id]["no_cache"][round][layer]

            print(f"Tensor shapes: {t_dyn.shape=} {t_ref.shape=}")
            
            if prefill:
                s=t_ref
            else:        
                s=t_ref[:,-1:,:]

            if torch.all(torch.eq(t_dyn,s)):
                result="EQUAL"
            else:
                result="NOT EQUAL"            

            print(f"{prefill_indicator}{round=} {layer=}: decoder intermediate result {result_id} (dynamic caching) {result} to corresponding column in intermediate result tensor without cache")


def check_decoder_internals():

    compare_decoder_result("input_layernorm")
    compare_decoder_result("self_attention")
    compare_decoder_result("attention_and_input") 
    compare_decoder_result("post_attention_layernorm")
    compare_decoder_result("norm_input")
    compare_decoder_result("rsqrt_input_pow")
    compare_decoder_result("rsqrt_input_mean")
    compare_decoder_result("rsqrt_input")
    compare_decoder_result("rsqrt_output")
    compare_decoder_result("norm_raw_output")


if __name__ == "__main__":
    # Step 1: Unpickle the class-level variables of GlobalVariables from a disk file
    load_global_variables()
    
    # Step 2: Perform analysis on the class-level variables
    hidden_state_checks()
    key_value_checks()   
    # compare the hidden state between dynamic caching and no cache  when the divergence first appears
    # that is, output_length=5, output of layer #6 = input to layer #7

    characterize_tensor_differences(GlobalVariables.tensor_dict["hidden_states"]["dynamic"][5][7], GlobalVariables.tensor_dict["hidden_states"]["no_cache"][5][7][:,-1:,:])   

    check_decoder_internals()

    print("Numeric differences in post_attention_layernorm (when divergence first occurs)")
    characterize_tensor_differences(GlobalVariables.tensor_dict["post_attention_layernorm"]["dynamic"][5][6], GlobalVariables.tensor_dict["post_attention_layernorm"]["no_cache"][5][6][:,-1:,:])   

    print("Numeric differences in mean calculatiob (within post_attention_layernorm) (when divergence first occurs)")
    characterize_tensor_differences(GlobalVariables.tensor_dict["rsqrt_input_mean"]["dynamic"][5][6], GlobalVariables.tensor_dict["rsqrt_input_mean"]["no_cache"][5][6][:,-1:,:])   
