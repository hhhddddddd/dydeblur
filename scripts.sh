#!/bin/bash

'''train
'''

python train.py -s data/D_NeRF/mutant -m output/dydeblur/D_NeRF/mutant --eval --is_blender # D_NeRF

'''render
'''

python train.py -s data/D_NeRF/mutant -m output/dydeblur/D_NeRF/mutant --eval --is_blender # D_NeRF

'''test
'''
# D_NeRF
python train.py -s data/D_NeRF/mutant -m output/dydeblur/D_NeRF/mutant --eval --is_blender # D_NeRF