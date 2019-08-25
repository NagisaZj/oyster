

#visulizing training curves
python ./viskit/frontend.py ./output/sparse-point-robot/first/

python ./viskit/frontend.py ./results/hard-code/2019_07_06_20_15_38/  ./results/hard-code/2019_07_06_20_16_12/

#running experiments
CUDA_VISIBLE_DEVICES=7 python launch_experiment.py ./configs/sparse-point-robot-show.json
CUDA_VISIBLE_DEVICES=0 python launch_experiment.py ./configs/ant-goal.json
CUDA_VISIBLE_DEVICES=0 python launch_experiment.py ./configs/sparse-point-robot-show-seed.json --gpu 1
CUDA_VISIBLE_DEVICES=0 python launch_experiment_explore.py ./configs/sparse-point-robot-show-exp.json --gpu 1
python launch_experiment.py ./configs/sparse-point-robot-show-atten.json --gpu 1
python launch_experiment.py ./configs/sparse-point-robot-show-snail.json --gpu 0
CUDA_VISIBLE_DEVICES=1 python launch_experiment.py ./configs/goal-pitfall.json
CUDA_VISIBLE_DEVICES=1 python launch_experiment.py ./configs/goal-pitfall-seed.json
python ./viskit/frontend.py ./output/sparse-point-robot/2019_07_07_09_19_18/  ./output/sparse-point-robot/2019_07_07_09_19_23/ --port=5000

python ./viskit/frontend.py ./output/sparse-point-robot/2019_07_07_09_19_18/  ./output/sparse-point-robot/2019_07_07_09_19_23/ ./SMMout/sparse-point-robot/2019_07_09_15_50_16/  ./SMMout/sparse-point-robot/2019_07_09_19_42_14/ ./SMMout/sparse-point-robot/2019_07_09_19_50_35/ ./SMMout/sparse-point-robot/2019_07_10_14_05_27/  --port=5005


python ./viskit/frontend.py ./output1/sparse-point-robot/2019_07_10_18_53_50  ./output1/sparse-point-robot/2019_07_10_18_54_11 ./output1/sparse-point-robot/2019_07_10_18_54_33  --port=5005

python ./viskit/frontend.py   /home/lthpc/Desktop/Research/Sampling/oyster/output1/ant-goal/2019_07_21_09_50_25       /home/lthpc/Desktop/Research/Sampling/oyster/output1/ant-goal/2019_07_21_10_01_21  --port=5005


python ./viskit/frontend.py   /home/lthpc/Desktop/Research/Sampling/oyster/output1/sparse-point-robot/2019_07_27_14_04_25       /home/lthpc/Desktop/Research/Sampling/oyster/output1/sparse-point-robot/2019_07_27_14_04_36  /home/lthpc/Desktop/Research/Sampling/oyster/output1/sparse-point-robot/2019_07_27_14_39_13 --port=5005

python ./viskit/frontend.py  ./output1/sparse-point-robot/2019_07_27_14_04_36  ./output1/sparse-point-robot/2019_07_28_15_13_39  ./output1/sparse-point-robot/2019_07_28_15_23_20   ./output1/sparse-point-robot/2019_07_19_15_10_26   ./output1/sparse-point-robot/2019_07_30_09_57_29  ./output1/sparse-point-robot/2019_08_13_11_27_10  ./output1/sparse-point-robot/2019_08_13_11_27_18  ./output1/sparse-point-robot/2019_08_21_09_15_52  ./output1/sparse-point-robot/2019_08_21_09_15_55 --port=5001

python ./viskit/frontend.py  /home/lthpc/Desktop/Research/Sampling/oyster/output1/goal-pitfall/2019_07_28_10_33_54  /home/lthpc/Desktop/Research/Sampling/oyster/output1/goal-pitfall/2019_07_28_22_02_20   --port=5002

python ./viskit/frontend.py ./outputa/sparse-point-robot/2019_08_24_16_15_02  ./outputa/sparse-point-robot/2019_08_24_16_15_09  ./outputa/sparse-point-robot/2019_08_24_16_15_18  ./output1/sparse-point-robot/2019_07_27_14_04_36 --port=5005