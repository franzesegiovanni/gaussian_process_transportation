import os 
import numpy as np
import matplotlib.pyplot as plt
import warnings
from models.model_gpt import Multiple_Reference_Frames_GPT
warnings.filterwarnings("ignore")
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
np.set_printoptions(precision=2) 

script_path = str(os.path.dirname(__file__))
filename = script_path + '/data/' + 'reach_target'
use_extra_points = True
policy=Multiple_Reference_Frames_GPT()
policy.load_dataset(filename, use_extra_points=use_extra_points)

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First subplot - Original dataset
ax1.grid(color='gray', linestyle='-', linewidth=1)
ax1.set_facecolor('white')
ax1.set_xlim(-60, 60)
ax1.set_ylim(-60, 60)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.grid(True)
ax1.set_title('Out-of-Distribution Goal Frame', fontsize=14)
source_index=2
for target_index in range(9):
    policy.reproduce(source_index, target_index, ax=ax1, compute_metrics=False)
for spine in ax1.spines.values():
    spine.set_linewidth(2)

# Test on a different dataset
filename = script_path  + '/data/reach_target_new'
frames_new = np.load(filename + '.npy', allow_pickle=True, encoding='latin1')[()]

### Coordinate systems transformation
A_test = frames_new['A']
b_test = frames_new['b']

policy.load_test_dataset(A_test, b_test, use_extra_points=use_extra_points)

# Second subplot - Out-of-distribution test
ax2.grid(color='gray', linestyle='-', linewidth=1)
ax2.set_facecolor('white')
ax2.set_xlim(-80, 60)
ax2.set_ylim(-80, 60)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.grid(True)
ax2.set_title('Out-of-Distribution Starting and Goal Frame', fontsize=14)
for i in range(9):
    policy.generalize(index_source=2, index_target=i, ax=ax2)
for spine in ax2.spines.values():
    spine.set_linewidth(2)

# Adjust layout and save as PDF
plt.tight_layout()
fig.savefig(script_path + '/figs/gpt_combined.pdf', dpi=300, bbox_inches='tight')



plt.show()



