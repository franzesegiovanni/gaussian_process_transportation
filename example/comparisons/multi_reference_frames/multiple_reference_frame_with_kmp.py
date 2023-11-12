import os 
import numpy as np
import matplotlib.pyplot as plt
import warnings
from models.model_kmp import Multiple_Reference_Frames_KMP as Multiple_Reference_Frames
warnings.filterwarnings("ignore")
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
np.set_printoptions(precision=2) 

script_path = str(os.path.dirname(__file__))
filename = script_path + '/data/' + 'reach_target'

policy=Multiple_Reference_Frames()
policy.load_dataset(filename)
fig, ax = plt.subplots()
ax.grid(color='gray', linestyle='-', linewidth=1)
# Customize the background color
ax.set_facecolor('white')
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(True)
ax.set_title('Kernelized Movement Primtives', fontsize=18)
source_index=2
for target_index in range(9):
    policy.reproduce(source_index, target_index, ax=ax, compute_metrics=False)
for spine in ax.spines.values():
    spine.set_linewidth(2)
#save figure
fig.savefig(script_path + '/figs/kmp.png', dpi=1200, bbox_inches='tight')

# Test on a differnet dataset
filename = script_path  + '/data/reach_target_new'

frames_new = np.load(filename + '.npy', allow_pickle=True, encoding='latin1')[()]

### Coordinate systems transformation
A_test = frames_new['A']
b_test = frames_new['b']

policy.load_test_dataset(A_test, b_test)

fig, ax = plt.subplots()
ax.grid(color='gray', linestyle='-', linewidth=1)
# Customize the background color
ax.set_facecolor('white')
ax.set_xlim(-80, 60)
ax.set_ylim(-80, 60)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(True)
# ax.set_title('GPT', fontsize=16)
for i in range(9):
    policy.generalize(index_source=2, index_target=i,  ax=ax)
for spine in ax.spines.values():
    spine.set_linewidth(2)
fig.savefig(script_path + '/figs/kmp_ood.png', dpi=1200, bbox_inches='tight')



plt.show()


