### command line to run experiments

export PYTHONPATH="${PYTHONPATH}:/root/code/gcsl_ant"

python RRT_star/Generate_graph.py ant16rooms
python RRT_star/Finetune_Vpolicy.py ant16rooms
python RRT_star/SupervisedLearning.py ant16rooms 1000 500
python RRT_star/TestLTLspecs_Buchi.py ant16rooms 10

nohup python RRT_star/Generate_graph.py ant16rooms > training.log &