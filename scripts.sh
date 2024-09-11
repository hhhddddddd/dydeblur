#!/bin/bash

'''train
'''

python train.py -s data/D_NeRF/trex -m output/dydeblur/D_NeRF/trex --eval --is_blender # D_NeRF

'''render
'''

# python render.py -m output/dydeblur/D_NeRF/trex --mode render # D_NeRF

'''test
'''
# D_NeRF
# python metrics.py -m output/dydeblur/D_NeRF/trex # D_NeRF