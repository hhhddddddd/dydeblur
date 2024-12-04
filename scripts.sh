#!/bin/bash

'''train
'''
# python train.py -s data/DyBluRF/stereo_blur_dataset/children/dense -m output/dydeblur/DyBluRF/children --eval --iterations 40000
# python train.py -s data/D_NeRF/trex -m output/dydeblur/D_NeRF/trex -o dynamic --eval --is_blender # D_NeRF

# python train.py -s data/D2RF/Camp -m output/dydeblur/D2RF/Camp -o new -e only_ssim0.8 -c -1 --eval --iterations 40000
# python train.py -s data/D2RF/Car -m output/dydeblur/D2RF/Car -o new -e only_ssim0.8 -c 0.01 --eval --iterations 40000
# python train.py -s data/D2RF/Dining1 -m output/dydeblur/D2RF/Dining1 -o new -e only_ssim0.8 -c 1.0 --eval --iterations 40000
# python train.py -s data/D2RF/Dining2 -m output/dydeblur/D2RF/Dining2 -o new -e only_ssim0.8 -c 0.1 --eval --iterations 40000
# python train.py -s data/D2RF/Dock -m output/dydeblur/D2RF/Dock -o new -e only_ssim0.8 --eval -c 0.9 --iterations 40000
# python train.py -s data/D2RF/Gate -m output/dydeblur/D2RF/Gate -o new -e only_ssim0.8 --eval -c -1 --iterations 40000
# python train.py -s data/D2RF/Mountain -m output/dydeblur/D2RF/Mountain -o new -e only_ssim0.8 -c -1 --eval --iterations 40000
# python train.py -s data/D2RF/Shop -m output/dydeblur/D2RF/Shop -o new -e only_ssim0.8 -c 1.0 --eval --iterations 40000


# python train.py -s data/DyBluRF/stereo_blur_dataset/basketball/dense -m output/dydeblur/DyBluRF/basketball -o new -c 1.0 -e only_mask0.001 --eval --iterations 40000
# python train.py -s data/DyBluRF/stereo_blur_dataset/children/dense -m output/dydeblur/DyBluRF/children -o new -c 1.0 -e only_mask0.001 --eval --iterations 40000
# python train.py -s data/DyBluRF/stereo_blur_dataset/sailor/dense -m output/dydeblur/DyBluRF/sailor -o new -c 1.0 -e only_mask0.001 --eval --iterations 40000
# python train.py -s data/DyBluRF/stereo_blur_dataset/seesaw/dense -m output/dydeblur/DyBluRF/seesaw -o new -c 1.0 -e only_mask0.001 --eval --iterations 40000
# python train.py -s data/DyBluRF/stereo_blur_dataset/skating/dense -m output/dydeblur/DyBluRF/skating -o new -c 1.0 -e only_mask0.001 --eval --iterations 40000
# python train.py -s data/DyBluRF/stereo_blur_dataset/street/dense -m output/dydeblur/DyBluRF/street -o new -c 1.0 -e only_mask0.001 --eval --iterations 40000


# python train.py -s data/DyBluRF/stereo_blur_dataset/sailor/dense -m output/dydeblur/DyBluRF/sailor -o point -e sfm --eval --iterations 40000
# python train.py -s data/DyBluRF/stereo_blur_dataset/sailor/dense -m output/dydeblur/DyBluRF/sailor -o point -e mvs --eval --iterations 40000
# python train.py -s data/DyBluRF/stereo_blur_dataset/sailor/dense -m output/dydeblur/DyBluRF/sailor -o point -e random --eval --iterations 40000


'''render
'''

# python render.py -m output/dydeblur/D_NeRF/trex -o dynamic -t 2024-09-19_22:27 --mode render
# python render.py -m output/dydeblur/D_NeRF/trex -o dynamic -t 2024-09-19_22:35 --mode render
# python render.py -m output/dydeblur/D_NeRF/trex -o dynamic -t 2024-09-19_21:16 --mode render
# python render.py -m output/dydeblur/D_NeRF/trex -o dynamic -t 2024-09-19_22:06 --mode render


# python render.py -m output/dydeblur/D2RF/Camp -o new -t 2024-11-26_16:27 -c -1 --mode render # 0.1
# python render.py -m output/dydeblur/D2RF/Car -o new -t 2024-11-26_16:58 -c 0.01 --mode render # 0.1
# python render.py -m output/dydeblur/D2RF/Dining1 -o new -t 2024-11-26_12:48 -c 1.0 --mode render # 0.9
# python render.py -m output/dydeblur/D2RF/Dining2 -o new -t 2024-11-26_11:39 -c 0.1 --mode render # true 0.1
# python render.py -m output/dydeblur/D2RF/Dock -o new -t 2024-11-28_21:54 -c 0.9 --mode render # 1.0
# python render.py -m output/dydeblur/D2RF/Gate -o new -t 2024-11-26_16:29 -c -1 --mode render # 0.1
# python render.py -m output/dydeblur/D2RF/Mountain -o new -t 2024-11-22_22:35 -c -1 --mode render # 0.1
# python render.py -m output/dydeblur/D2RF/Shop -o new -t 2024-11-26_13:37 -c 1.0 --mode render # 1.0


# python render.py -m output/dydeblur/DyBluRF/sailor -o test -t 2024-09-28_23:10 --mode render 
# python render.py -m output/dydeblur/DyBluRF/basketball -o test -t 2024-09-29_11:22 --mode render 
# python render.py -m output/dydeblur/DyBluRF/sailor -o test -t 2024-09-28_23:10 --mode render 
# python render.py -m output/dydeblur/DyBluRF/seesaw -o test -t 2024-09-27_20:44 --mode render 

'''test
'''

# python metrics.py -m output/dydeblur/D_NeRF/trex 


python metrics.py -s data/D2RF/Camp -m output/dydeblur/D2RF/Camp
python metrics.py -s data/D2RF/Car -m output/dydeblur/D2RF/Car
python metrics.py -s data/D2RF/Dining1 -m output/dydeblur/D2RF/Dining1
python metrics.py -s data/D2RF/Dining2 -m output/dydeblur/D2RF/Dining2
python metrics.py -s data/D2RF/Dock -m output/dydeblur/D2RF/Dock
python metrics.py -s data/D2RF/Gate -m output/dydeblur/D2RF/Gate
python metrics.py -s data/D2RF/Mountain -m output/dydeblur/D2RF/Mountain
python metrics.py -s data/D2RF/Shop -m output/dydeblur/D2RF/Shop


# python metrics.py -m output/dydeblur/DyBluRF/sailor 
# python metrics.py -m output/dydeblur/DyBluRF/basketball 
# python metrics.py -m output/dydeblur/DyBluRF/street 
# python metrics.py -m output/dydeblur/DyBluRF/seesaw

