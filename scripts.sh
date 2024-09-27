#!/bin/bash

'''train
'''
# python train.py -s data/DyBluRF/stereo_blur_dataset/children/dense -m output/dydeblur/DyBluRF/children --eval --iterations 40000
# python train.py -s data/D_NeRF/trex -m output/dydeblur/D_NeRF/trex -o dynamic --eval --is_blender # D_NeRF

# python train.py -s data/D2RF/Shop -m output/dydeblur/D2RF/Shop -o new --eval --iterations 40000
# python train.py -s data/D2RF/Dining1 -m output/dydeblur/D2RF/Dining1 -o new --eval --iterations 40000 
# python train.py -s data/D2RF/Camp -m output/dydeblur/D2RF/Camp -o new --eval --iterations 40000
# python train.py -s data/D2RF/Gate -m output/dydeblur/D2RF/Gate -o new --eval --iterations 40000 
# python train.py -s data/D2RF/Mountain -m output/dydeblur/D2RF/Mountain -o new --eval --iterations 40000
# python train.py -s data/D2RF/Car -m output/dydeblur/D2RF/Car -o new --eval --iterations 40000

python train.py -s data/D2RF/Dining2 -m output/dydeblur/D2RF/Dining2 -o new --eval --iterations 40000 
python train.py -s data/D2RF/Dock -m output/dydeblur/D2RF/Dock -o new --eval --iterations 40000


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
dining2: gaussian NaN; only 3 gaussians;                                mideo resuts
dock: gaussian NaN 4550 iter; only 3 gaussians                          mideo result
gate: gaussian NaN 3000 iter: dynamic nan, mask_loss nan;               over fitting                        ok 260000

mountain: gaussian always unchanged -> zero (3100 iter); ssim is low    grad is xiaoshi                     20 gaussian 

'''

'''render
'''

# python render.py -m output/dydeblur/D_NeRF/trex -o dynamic -t 2024-09-19_22:27 --mode render # D_NeRF
# python render.py -m output/dydeblur/D_NeRF/trex -o dynamic -t 2024-09-19_22:35 --mode render # D_NeRF
# python render.py -m output/dydeblur/D_NeRF/trex -o dynamic -t 2024-09-19_21:16 --mode render # D_NeRF
# python render.py -m output/dydeblur/D_NeRF/trex -o dynamic -t 2024-09-19_22:06 --mode render # D_NeRF

'''test
'''

# python metrics.py -m output/dydeblur/D_NeRF/trex # D_NeRF
