import os 
import numpy as np
import matplotlib.pyplot as plt
import warnings
from models.model_laplacian_editing import Multiple_Reference_Frames_LA
warnings.filterwarnings("ignore")
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
np.set_printoptions(precision=2) 

script_path = str(os.path.dirname(__file__))
filename = script_path + '/data/' + 'reach_target'
use_extra_points = True
policy=Multiple_Reference_Frames_LA()
policy.load_dataset(filename, use_extra_points=use_extra_points)
fig, ax = plt.subplots()
ax.grid(color='gray', linestyle='-', linewidth=1)
# Customize the background color
ax.set_facecolor('white')
ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(True)
ax.set_title('Laplacian Editing', fontsize=18)
source_index=2
for target_index in range(9):
    policy.reproduce(source_index, target_index, ax=ax, compute_metrics=False)
for spine in ax.spines.values():
    spine.set_linewidth(2)
#save figure
fig.savefig(script_path + '/figs/la_dataset_5_key_points.png', dpi=1200, bbox_inches='tight')

# Test on a differnet dataset
filename = script_path  + '/data/reach_target_new'

frames_new = np.load(filename + '.npy', allow_pickle=True, encoding='latin1')[()]

### Coordinate systems transformation
A_test = frames_new['A']
b_test = frames_new['b']

policy.load_test_dataset(A_test, b_test, use_extra_points=use_extra_points)

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
    policy.generalize(index_source=source_index, index_target=i,  ax=ax)
for spine in ax.spines.values():
    spine.set_linewidth(2)
fig.savefig(script_path + '/figs/la_ood_5_key_points.png', dpi=1200, bbox_inches='tight')



plt.show()



