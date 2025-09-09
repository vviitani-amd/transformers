import numpy as np

file_path='generated_scores_unknown.npy'
# Load the 2D numpy array from the file
scores_nocache = np.load(file_path)

file_path='generated_scores_dynamic.npy'
# Load the 2D numpy array from the file
scores_dynamic = np.load(file_path)

file_path='generated_scores_static.npy'
# Load the 2D numpy array from the file
scores_static = np.load(file_path)

#first differing newly generated token between dynamic and reference

first_different_token_dynamic=12
token_reference=1476
token_dynamic=1144

print("dynamic vs reference")
print(f"{scores_nocache[first_different_token_dynamic,token_reference]=} {scores_nocache[first_different_token_dynamic,token_dynamic]=}")
print(f"{scores_dynamic[first_different_token_dynamic,token_reference]=} {scores_dynamic[first_different_token_dynamic,token_dynamic]=}")

first_different_token_static=3
token_reference=573
token_static=235265

print("static vs reference")
print(f"{scores_nocache[first_different_token_static,token_reference]=} {scores_nocache[first_different_token_static,token_static]=}")
print(f"{scores_static[first_different_token_static,token_reference]=} {scores_static[first_different_token_static,token_static]=}")


# # Print the shape of the array
# print("Shape of the scores array:", scores_array.shape)

# # Print the type of the array elements
# print("Data type of array elements:", scores_array.dtype)

# # # Print a sample of the array contents (first 5 rows)
# # print("Sample of the scores (first 5 rows):")
# # print(scores_array[:5])

# # Add any additional inspections needed
# # For example, statistics or specific value checks
# print("Statistics of the scores array:")
# print("Mean:", np.mean(scores_array))
# print("Standard Deviation:", np.std(scores_array))

