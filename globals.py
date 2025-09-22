class GlobalVariables:
    cache_id:str = ""
    cur_len:int = -1
    hidden_states=None
    
    key_states_precache=None
    value_states_precache=None
    key_states_postcache=None
    value_states_postcache=None
    
    output_length_cutoff:int = 5
   