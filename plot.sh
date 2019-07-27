

#visulizing training curves
python ./viskit/frontend.py ./output/sparse-point-robot/first/

python ./viskit/frontend.py ./results/hard-code/2019_07_06_20_15_38/  ./results/hard-code/2019_07_06_20_16_12/

#running experiments
CUDA_VISIBLE_DEVICES=7 python launch_experiment.py ./configs/sparse-point-robot-show.json
CUDA_VISIBLE_DEVICES=0 python launch_experiment.py ./configs/ant-goal.json

python ./viskit/frontend.py ./output/sparse-point-robot/2019_07_07_09_19_18/  ./output/sparse-point-robot/2019_07_07_09_19_23/ --port=5000

python ./viskit/frontend.py ./output/sparse-point-robot/2019_07_07_09_19_18/  ./output/sparse-point-robot/2019_07_07_09_19_23/ ./SMMout/sparse-point-robot/2019_07_09_15_50_16/  ./SMMout/sparse-point-robot/2019_07_09_19_42_14/ ./SMMout/sparse-point-robot/2019_07_09_19_50_35/ ./SMMout/sparse-point-robot/2019_07_10_14_05_27/  --port=5005


python ./viskit/frontend.py ./output1/sparse-point-robot/2019_07_10_18_53_50  ./output1/sparse-point-robot/2019_07_10_18_54_11 ./output1/sparse-point-robot/2019_07_10_18_54_33  --port=5005

python ./viskit/frontend.py   /home/lthpc/Desktop/Research/Sampling/oyster/output1/ant-goal/2019_07_21_09_50_25       /home/lthpc/Desktop/Research/Sampling/oyster/output1/ant-goal/2019_07_21_10_01_21  --port=5005