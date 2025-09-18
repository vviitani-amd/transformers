import numpy as np

# file_path='generated_scores_unknown.npy'
# # Load the 2D numpy array from the file
# scores_nocache = np.load(file_path)

# file_path='generated_scores_dynamic.npy'
# # Load the 2D numpy array from the file
# scores_dynamic = np.load(file_path)

# file_path='generated_scores_static.npy'
# # Load the 2D numpy array from the file
# scores_static = np.load(file_path)

# H100

# id1="unknown"
# id2="h100_unknown"

# print(f'Comparing score distributions {id1} vs {id2}')
# print("MI300 reference vs H100 reference")

# id1="dynamic"
# id2="h100_unknown"

# print(f'Comparing score distributions {id1} vs {id2}')
# print("MI300 dynamic vs H100 reference")



# id1="unknown"
# id2="h100_dynamic"

# print(f'Comparing score distributions {id1} vs {id2}')
# print("MI300 reference vs H100 dynamic")

id1="dynamic"
id2="h100_dynamic"

print(f'Comparing score distributions {id1} vs {id2}')
print("MI300 dynamic vs H100 dynamic")


csv_lines=[f'round,"Number of correctly ranked tokens","Percentage of non-zero score differences (all tokens)","Mean difference (all tokens)']
csv_lines[0] += f',"Percentage of non-zero score differences (top-K tokens)","Mean difference (top-K tokens)"'


file_path=f'generated_scores_{id1}.npy'
scores1 = np.load(file_path)

file_path=f'generated_scores_{id2}.npy'
scores2 = np.load(file_path)


#first differing newly generated token between dynamic and reference

for round in range(20):
    diff=scores2[round,:]-scores1[round,:]
    vocab_size=len(diff)

    ranking1=np.argsort(scores1[round,:])
    ranking2=np.argsort(scores2[round,:])


    num_correctly_ranked=0
    while scores2[round,ranking1[num_correctly_ranked]] == scores2[round,ranking2[num_correctly_ranked]]:
        num_correctly_ranked += 1
        if num_correctly_ranked >= vocab_size:
            break

    nonzero_percentage=100*np.count_nonzero(diff) / vocab_size
    mean_absdiff = np.mean(np.abs(diff))

    if num_correctly_ranked > 0:
        nonzero_percentage_top=100*np.count_nonzero(diff[:num_correctly_ranked]) / num_correctly_ranked
        mean_absdiff_top = np.mean(np.abs(diff[:num_correctly_ranked]))
    else:
        nonzero_percentage_top=-1
        mean_absdiff_top = -1

    csv_lines += [f'{round},{num_correctly_ranked},{nonzero_percentage},{mean_absdiff},{nonzero_percentage_top},{mean_absdiff_top}']   

for line in csv_lines:
    print(line)             