#!/bin/bash

# d2rf="Camp Car Dining1 Dining2 Dock Gate Mountain Shop"
# dyblurf="man seesaw skating street third women"
d2rf="Camp"
dyblurf="women"
skating="skating"
K="3 5 7 9 11 13 15"
declare -A sc
sc["Camp"]="0.1"
sc["Car"]="0.01"
sc["Dining1"]="0.1"
sc["Dining2"]="0.01"
sc["Dock"]="0.01"
sc["Gate"]="0.01"
sc["Mountain"]="0.01"
sc["Shop"]="0.1"

sc["man"]="0.01"
sc["seesaw"]="0.04"
sc["skating"]="0.01"
sc["street"]="0.01"
sc["third"]="0.02"
sc["women"]="0.01"

declare -A time
time["Camp"]="2025-02-17_00:23"
time["Car"]="2025-02-17_00:23"
time["Dining1"]="2025-02-17_00:23"
time["Dining2"]="2025-02-17_00:23"
time["Dock"]="2025-02-17_00:23"
time["Gate"]="2025-02-17_00:23"
time["Mountain"]="2025-02-17_00:23"
time["Shop"]="2025-02-17_00:23"

time["man"]="2025-02-17_00:23"
time["seesaw"]="2025-02-17_00:23"
time["skating"]="2025-02-17_00:23"
time["street"]="2025-02-17_00:23"
time["third"]="2025-02-17_00:23"
time["women"]="2025-02-17_00:23"

export CUDA_VISIBLE_DEVICES=7
for k in $K; do
    python train.py -s /home/xuankai/dataset/xuankai/new_deblur4dgs/women/x2.5 -m output/dydeblur/Dydeblur/dyblurf/women \
                    -o new -c 0.01 -e l_k_$k -w 0.0 -k $k --eval --iterations 40000
done

###------------------------------ train low ------------------------------###
# export CUDA_VISIBLE_DEVICES=6
# for scene in $dyblurf; do
#     python train.py -s /home/xuankai/dataset/xuankai/new_deblur4dgs/$scene/x2.5 -m output/dydeblur/Dydeblur/dyblurf/$scene \
#                     -o new -c ${sc["$scene"]} -e l_sc${sc["$scene"]}_noaddpoint_noscaleloss -w 0.0 --eval --iterations 40000
# done

# export CUDA_VISIBLE_DEVICES=1
# for scene in $d2rf; do
#     python train.py -s /home/xuankai/dataset/new_d2rf_test/$scene/x2 -m output/dydeblur/Dydeblur/d2rf/$scene \
#                     -o new -c ${sc["$scene"]} -e l_sc${sc["$scene"]}_nodownnoaddpoint_nodmask --not_use_dynamic_mask --eval --iterations 40000
# done

# export CUDA_VISIBLE_DEVICES=3
# for scene in $dyblurf; do
#     python train.py -s /home/xuankai/dataset/xuankai/new_deblur4dgs/$scene/x2.5 -m output/dydeblur/Dydeblur/dyblurf/$scene \
#                     -o new -c ${sc["$scene"]} -e l_sc${sc["$scene"]}_nodownnoaddpoint_dmask --eval --iterations 40000
# done

# export CUDA_VISIBLE_DEVICES=4
# for scene in $dyblurf; do
#     python train.py -s /home/xuankai/dataset/xuankai/new_deblur4dgs/$scene/x2.5 -m output/dydeblur/Dydeblur/dyblurf/$scene \
#                     -o new -c ${sc["$scene"]} -e l_sc${sc["$scene"]}_nodownnoaddpoint_dembmask --use_emb_dynamic_mask --eval --iterations 40000
# done

# export CUDA_VISIBLE_DEVICES=5
# for scene in $skating; do
#     python train.py -s /home/xuankai/dataset/new_dyblurf/$scene/x2.5 -m output/dydeblur/Dydeblur/dyblurf/$scene \
#                     -o new -c ${sc["$scene"]} -e l_sc${sc["$scene"]}_masksmoothdown_dyblurf --eval --iterations 40000
# done


###------------------------------ train high ------------------------------###
# export CUDA_VISIBLE_DEVICES=0
# for scene in $dyblurf; do
#     python train.py -s /home/xuankai/dataset/xuankai/new_deblur4dgs/$scene/x1 -m output/dydeblur/Dydeblur/dyblurfh/$scene \
#                     -o new -c ${sc["$scene"]} -e h_sc${sc["$scene"]}_masksmoothdown --eval --iterations 40000
# done

# export CUDA_VISIBLE_DEVICES=6
# for scene in $d2rf; do
#     python train.py -s /home/xuankai/dataset/new_d2rf_test/$scene/x1 -m output/dydeblur/Dydeblur/d2rfh/$scene \
#                     -o new -c ${sc["$scene"]} -e h_sc${sc["$scene"]}_masksmoothdown --eval --iterations 40000
# done

# export CUDA_VISIBLE_DEVICES=5
# for scene in $skating; do
#     python train.py -s /home/xuankai/dataset/new_dyblurf/$scene/x1 -m output/dydeblur/Dydeblur/dyblurfh/$scene \
#                     -o new -c ${sc["$scene"]} -e h_sc${sc["$scene"]}_masksmoothdown_dyblurf --eval --iterations 40000
# done

###------------------------------ render low ------------------------------###
# export CUDA_VISIBLE_DEVICES=5
# for scene in $dyblurf; do
#     python render.py -m output/dydeblur/Dydeblur/dyblurf/$scene \
#                      -o new -c ${sc["$scene"]} -t ${time["$scene"]} --mode render 
# done

# for scene in $d2rf; do
#     python render.py -m output/dydeblur/Dydeblur/d2rf/$scene \
#                      -o new -c ${sc["$scene"]} -t ${time["$scene"]} --mode render 
# done

###------------------------------ render high ------------------------------###
# export CUDA_VISIBLE_DEVICES=5
# for scene in $dyblurf; do
#     python render.py -m output/dydeblur/Dydeblur/dyblurfh/$scene \
#                      -o new -c ${sc["$scene"]} -t ${time["$scene"]} --mode render 
# done

# for scene in $d2rf; do
#     python render.py -m output/dydeblur/Dydeblur/d2rfh/$scene \
#                      -o new -c ${sc["$scene"]} -t ${time["$scene"]} --mode render 
# done

###------------------------------ metrics low ------------------------------###
# export CUDA_VISIBLE_DEVICES=5
# for scene in $dyblurf; do
#     python metrics.py -s /home/xuankai/dataset/xuankai/new_deblur4dgs/$scene/x2.5 -m output/dydeblur/Dydeblur/dyblurf/$scene --use_alex
# done

# for scene in $d2rf; do
#     python metrics.py -s /home/xuankai/dataset/new_d2rf_test/$scene/x2 -m output/dydeblur/Dydeblur/d2rf/$scene --use_alex
# done

# for scene in $skating; do
#     python metrics.py -s -s /home/xuankai/dataset/new_dyblurf/$scene/x2.5 -m output/dydeblur/Dydeblur/dyblurf/$scene --use_alex
# done

###------------------------------ metrics high ------------------------------###
# export CUDA_VISIBLE_DEVICES=5
# for scene in $dyblurf; do
#     python metrics.py -s /home/xuankai/dataset/xuankai/new_deblur4dgs/$scene/x1 -m output/dydeblur/Dydeblur/dyblurfh/$scene --use_alex
# done

# for scene in $d2rf; do
#     python metrics.py -s /home/xuankai/dataset/new_d2rf_test/$scene/x1 -m output/dydeblur/Dydeblur/d2rfh/$scene --use_alex
# done

# for scene in $skating; do
#     python metrics.py -s -s /home/xuankai/dataset/new_dyblurf/$scene/x1 -m output/dydeblur/Dydeblur/dyblurfh/$scene --use_alex
# done