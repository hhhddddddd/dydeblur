#!/bin/bash

'''train
'''
python train.py -s data/DyBluRF/stereo_blur_dataset/children/dense -m output/dydeblur/DyBluRF/children --eval --iterations 40000
# python train.py -s data/D_NeRF/trex -m output/dydeblur/D_NeRF/trex -o test --eval --is_blender # D_NeRF

'''render
'''

# python render.py -m output/dydeblur/D_NeRF/trex --mode render # D_NeRF

'''test
'''
# D_NeRF
# python metrics.py -m output/dydeblur/D_NeRF/trex # D_NeRF