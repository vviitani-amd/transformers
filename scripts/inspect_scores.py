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

csv_lines=[f'round,"Number of correctly ranked tokens","Percentage of non-zero score differences (all tokens)","Mean difference (all tokens)']
csv_lines[0] += f',"Percentage of non-zero score differences (top-K tokens)","Mean difference (top-K tokens)"'

print("Score distribution on H100")

file_path='generated_scores_h100_unknown.npy'
scores_nocache = np.load(file_path)

file_path='generated_scores_h100_dynamic.npy'
scores_dynamic = np.load(file_path)

file_path='generated_scores_h100_static.npy'
scores_static = np.load(file_path)


#first differing newly generated token between dynamic and reference

for round in range(20):
    diff=scores_dynamic[round,:]-scores_nocache[round,:]
    vocab_size=len(diff)

    ranking_nocache=np.argsort(scores_nocache[round,:])
    ranking_dynamic=np.argsort(scores_dynamic[round,:])


    num_correctly_ranked=0
    while scores_dynamic[round,ranking_nocache[num_correctly_ranked]] == scores_dynamic[round,ranking_dynamic[num_correctly_ranked]]:
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