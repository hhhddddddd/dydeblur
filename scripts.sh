#!/bin/bash

'''train
'''
# python train.py -s data/DyBluRF/stereo_blur_dataset/children/dense -m output/dydeblur/DyBluRF/children --eval --iterations 40000
# python train.py -s data/D_NeRF/trex -m output/dydeblur/D_NeRF/trex -o dynamic --eval --is_blender # D_NeRF

# python train.py -s data/D2RF/Shop -m output/dydeblur/D2RF/Shop -o knn --eval --iterations 40000
# python train.py -s data/D2RF/Dining1 -m output/dydeblur/D2RF/Dining1 -o knn --eval --iterations 40000 
# python train.py -s data/D2RF/Camp -m output/dydeblur/D2RF/Camp -o knn --eval --iterations 40000
# python train.py -s data/D2RF/Gate -m output/dydeblur/D2RF/Gate -o knn --eval --iterations 40000 
# python train.py -s data/D2RF/Mountain -m output/dydeblur/D2RF/Mountain -o knn --eval --iterations 40000
# python train.py -s data/D2RF/Car -m output/dydeblur/D2RF/Car -o knn --eval --iterations 40000
# python train.py -s data/D2RF/Dining2 -m output/dydeblur/D2RF/Dining2 -o knn --eval --iterations 40000 
# python train.py -s data/D2RF/Dock -m output/dydeblur/D2RF/Dock -o knn --eval --iterations 40000


python train.py -s data/DyBluRF/stereo_blur_dataset/basketball/dense -m output/dydeblur/DyBluRF/basketball -o test --eval --iterations 40000
python train.py -s data/DyBluRF/stereo_blur_dataset/children/dense -m output/dydeblur/DyBluRF/children -o test --eval --iterations 40000
python train.py -s data/DyBluRF/stereo_blur_dataset/sailor/dense -m output/dydeblur/DyBluRF/sailor -o test --eval --iterations 40000
python train.py -s data/DyBluRF/stereo_blur_dataset/seesaw/dense -m output/dydeblur/DyBluRF/seesaw -o test --eval --iterations 40000
python train.py -s data/DyBluRF/stereo_blur_dataset/skating/dense -m output/dydeblur/DyBluRF/skating -o test --eval --iterations 40000
python train.py -s data/DyBluRF/stereo_blur_dataset/street/dense -m output/dydeblur/DyBluRF/street -o test --eval --iterations 40000


'''
shop: ok                                                                OOM: 90000 gaussian 34246 iter      ok  150000
dining1: ok                                                             OOM: only 30000 gaussians           ok 100000 

camp: OOM;                                                                                                  127 gaussian
car: gaussian NaN 3000 iter; only 3 gaussians;                          bad results                         6 gaussian
dining2: gaussian NaN; only 3 gaussians;                                mideo resuts                        ok 150000
dock: gaussian NaN 4550 iter; only 3 gaussians                          mideo result                        ok 130000
gate: gaussian NaN 3000 iter: dynamic nan, mask_loss nan;               over fitting                        ok 260000

mountain: gaussian always unchanged -> zero (3100 iter); ssim is low    grad is xiaoshi                     20 gaussian 


basketball:     3000 iter gaussian from 18000 -> 55 (ok)            test ok
children:       3000 iter gaussian from 50000 -> 15                 test gaussian stop change
sailor:         ok 3000 iter gaussian from 30000 -> 1000            test ok
seesaw:         ok 3000 iter gaussian from 110000 -> 30000
skating:        3000 iter gaussian from 65000 -> 1000
street:         3000 iter gaussian from 57000 -> 0

'''

'''render
'''

# python render.py -m output/dydeblur/D_NeRF/trex -o dynamic -t 2024-09-19_22:27 --mode render
# python render.py -m output/dydeblur/D_NeRF/trex -o dynamic -t 2024-09-19_22:35 --mode render
# python render.py -m output/dydeblur/D_NeRF/trex -o dynamic -t 2024-09-19_21:16 --mode render
# python render.py -m output/dydeblur/D_NeRF/trex -o dynamic -t 2024-09-19_22:06 --mode render

python render.py -m output/dydeblur/D2RF/Dining1 -o new -t 2024-09-27_08:13 --mode render
python render.py -m output/dydeblur/D2RF/Shop -o new -t 2024-09-26_23:56 --mode render
python render.py -m output/dydeblur/D2RF/Gate -o new -t 2024-09-26_19:36 --mode render
# python render.py -m output/dydeblur/D2RF/Dock -o new -t 2024-09-26_23:56 --mode render
# python render.py -m output/dydeblur/D2RF/Dining2 -o new -t 2024-09-26_23:56 --mode render

python render.py -m output/dydeblur/DyBluRF/sailor -o test -t 2024-09-28_23:10 --mode render 
python render.py -m output/dydeblur/DyBluRF/basketball -o test -t 2024-09-29_11:22 --mode render 
python render.py -m output/dydeblur/DyBluRF/street -o test -t 2024-09-28_15:15 --mode render 

'''test
'''

# python metrics.py -m output/dydeblur/D_NeRF/trex 

python metrics.py -m output/dydeblur/D2RF/Dining1
python metrics.py -m output/dydeblur/D2RF/Shop
python metrics.py -m output/dydeblur/D2RF/Gate

python metrics.py -m output/dydeblur/DyBluRF/sailor 
python metrics.py -m output/dydeblur/DyBluRF/basketball 
python metrics.py -m output/dydeblur/DyBluRF/street 

