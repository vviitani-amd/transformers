import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.testing_utils import (
    torch_device,
)


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
        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(selected_input, return_tensors="pt", padding=True).to(torch_device)
        print(f"Tokenized input: {input}")


        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_default=output
        output_text_cache = tokenizer.batch_decode(output, skip_special_tokens=True)
        output_difference = output

        # repeat the same with no caching
        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(selected_input, return_tensors="pt", padding=True).to(torch_device)
        
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False, use_cache=False)
        output_nocache=output
        output_text_reference = tokenizer.batch_decode(output, skip_special_tokens=True)
        output_difference -= output
        
        #test if we can use static cache implementation as reference

        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(selected_input, return_tensors="pt", padding=True).to(torch_device)
        
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False, cache_implementation="static")
        output_static = output
        output_text_static = tokenizer.batch_decode(output, skip_special_tokens=True)
        output_difference -= output
        


        print(f"Text generated with default KV-cache: {output_text_cache}")
        print(f"Text generated with no KV-cache: {output_text_reference}")
        print(f"Text generated with static KV-cache: {output_text_static}")
        

        print(f"{output_default=}")
        print(f"{output_nocache=}")
        print(f"{output_static=}")
        

        # print(f"Statistics of numerical difference with and without cache:")

        # # Convert the tensor to a floating point type
        # output_difference = output_difference.float()
        # # Calculate statistics
        # mean_value = torch.mean(output_difference)
        # std_dev = torch.std(output_difference)
        # min_value = torch.min(output_difference)
        # max_value = torch.max(output_difference)
        # median_value = torch.median(output_difference)
        # max_abs_value = torch.max(torch.abs(output_difference))
        # zero_count = torch.sum(output_difference == 0).item()
        # zero_fraction = zero_count / output_difference.numel()
        # tensor_shape = output_difference.shape
        # # Print statistics
        # print(f"Standard Deviation: {std_dev.item()}")
        # print(f"Maximum Absolute Difference: {max_abs_value.item()}")
        # print(f"Count of Zeros: {zero_count}")
        # print(f"Fraction of Zeros: {zero_fraction:.4f}")
        # print(f"Tensor Shape: {tensor_shape}")
        

if __name__ == "__main__":
    test_model_7b_fp16_modified()
