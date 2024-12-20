#!/bin/bash

'''train
'''

export CUDA_VISIBLE_DEVICES=3
# python train.py -s data/D2RF/Camp -m output/dydeblur/D2RF/Camp -o new -e mask_extent_skip_connect_align_sigmod_center5. -c -1 --eval --iterations 40000 # 
# python train.py -s data/D2RF/Car -m output/dydeblur/D2RF/Car -o new -e mask_align_depth -c 0.01 --eval --iterations 40000
# python train.py -s data/D2RF/Dining1 -m output/dydeblur/D2RF/Dining1 -o new -e mask_extent_skip_connect_align_sigmod_center5. -c 1.0 --eval --iterations 40000 # 
# python train.py -s data/D2RF/Dining2 -m output/dydeblur/D2RF/Dining2 -o new -e mask0.0005_extent_skip_connect_align_sigmod_center5. -c 0.1 --eval --iterations 40000
# python train.py -s data/D2RF/Dock -m output/dydeblur/D2RF/Dock -o new -e mask_align_depth --eval -c 0.9 --iterations 40000
# python train.py -s data/D2RF/Gate -m output/dydeblur/D2RF/Gate -o new -e mask_align_depth --eval -c -1 --iterations 40000 # 
# python train.py -s data/D2RF/Mountain -m output/dydeblur/D2RF/Mountain -o new -e mask0.00001_alig0.00001_skip_connect -c -1 --eval --iterations 40000
# python train.py -s data/D2RF/Shop -m output/dydeblur/D2RF/Shop -o new -e mask_extent_skip_connect_align0.0001_sigmod_center5._depth -c 1.0 --eval --iterations 40000


python train.py -s data/DyBluRF/stereo_blur_dataset/basketball/dense -m output/dydeblur/DyBluRF/basketball -o new -c 0.05 -e mask_align_extent --eval --iterations 40000
# python train.py -s data/DyBluRF/stereo_blur_dataset/children/dense -m output/dydeblur/DyBluRF/children -o new -c 0.006 -e mask_align_extent --eval --iterations 40000
# python train.py -s data/DyBluRF/stereo_blur_dataset/sailor/dense -m output/dydeblur/DyBluRF/sailor -o new -c 0.003 -e mask_align_extent --eval --iterations 40000
# python train.py -s data/DyBluRF/stereo_blur_dataset/seesaw/dense -m output/dydeblur/DyBluRF/seesaw -o new -c 0.04 -e mask_align_extent7.5  --eval --iterations 40000
# python train.py -s data/DyBluRF/stereo_blur_dataset/skating/dense -m output/dydeblur/DyBluRF/skating -o new -c 0.03 -e mask_align_extent5. --eval --iterations 40000
# python train.py -s data/DyBluRF/stereo_blur_dataset/street/dense -m output/dydeblur/DyBluRF/street -o new -c 0.03 -e mask_align_extent --eval --iterations 40000


'''render
'''

# python render.py -m output/dydeblur/D2RF/Camp -o new -t 2024-11-26_16:27 -c -1 --mode render # 0.1
# python render.py -m output/dydeblur/D2RF/Car -o new -t 2024-12-08_15:00 -c 0.01 --mode render # 0.1
# python render.py -m output/dydeblur/D2RF/Dining1 -o new -t 2024-11-26_12:48 -c 1.0 --mode render # 0.9
# python render.py -m output/dydeblur/D2RF/Dining2 -o new -t 2024-11-26_11:39 -c 0.1 --mode render # true 0.1
# python render.py -m output/dydeblur/D2RF/Dock -o new -t 2024-12-05_12:52 -c 0.9 --mode render # 1.0
# python render.py -m output/dydeblur/D2RF/Gate -o new -t 2024-11-26_16:29 -c -1 --mode render # 0.1
# python render.py -m output/dydeblur/D2RF/Mountain -o new -t 2024-11-22_22:35 -c -1 --mode render # 0.1
# python render.py -m output/dydeblur/D2RF/Shop -o new -t 2024-11-26_13:37 -c 1.0 --mode render # 1.0


# python render.py -m output/dydeblur/DyBluRF/basketball -o new -t 2024-09-29_11:22 --mode render 
# python render.py -m output/dydeblur/DyBluRF/children -o new -t 2024-09-29_11:22 --mode render 
# python render.py -m output/dydeblur/DyBluRF/sailor -o new -t 2024-09-28_23:10 --mode render 
# python render.py -m output/dydeblur/DyBluRF/seesaw -o new -t 2024-12-16_10:15 -c -1 --mode render 
# python render.py -m output/dydeblur/DyBluRF/skating -o new -t 2024-09-27_20:44 --mode render 
# python render.py -m output/dydeblur/DyBluRF/street -o new -t 2024-09-27_20:44 --mode render 

'''test
'''

# python metrics.py -s data/D2RF/Camp -m output/dydeblur/D2RF/Camp --use_alex
# python metrics.py -s data/D2RF/Car -m output/dydeblur/D2RF/Car --use_alex
# python metrics.py -s data/D2RF/Dining1 -m output/dydeblur/D2RF/Dining1 --use_alex
# python metrics.py -s data/D2RF/Dining2 -m output/dydeblur/D2RF/Dining2 --use_alex
# python metrics.py -s data/D2RF/Dock -m output/dydeblur/D2RF/Dock --use_alex
# python metrics.py -s data/D2RF/Gate -m output/dydeblur/D2RF/Gate --use_alex
# python metrics.py -s data/D2RF/Mountain -m output/dydeblur/D2RF/Mountain --use_alex
# python metrics.py -s data/D2RF/Shop -m output/dydeblur/D2RF/Shop --use_alex


# python metrics.py -s data/DyBluRF/stereo_blur_dataset/basketball/dense -m output/dydeblur/DyBluRF/basketball 
# python metrics.py -s data/DyBluRF/stereo_blur_dataset/children/dense -m output/dydeblur/DyBluRF/children
# python metrics.py -s data/DyBluRF/stereo_blur_dataset/sailor/dense -m output/dydeblur/DyBluRF/sailor 
# python metrics.py -s data/DyBluRF/stereo_blur_dataset/seesaw/dense -m output/dydeblur/DyBluRF/seesaw
# python metrics.py -s data/DyBluRF/stereo_blur_dataset/skating/dense -m output/dydeblur/DyBluRF/skating
# python metrics.py -s data/DyBluRF/stereo_blur_dataset/street/dense -m output/dydeblur/DyBluRF/street 
