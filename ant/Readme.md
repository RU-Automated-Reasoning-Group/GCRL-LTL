### command line to run experiments

export PYTHONPATH="${PYTHONPATH}:/root/code/gcsl_pseudo_algo/ant"

require installment of ltl2ba

export PATH=$PATH:/root/code/gcsl_pseudo_algo/ant/ltl2ba-1.2b1

python experiments/Generate_graph.py ant16rooms
python experiments/Finetune_Vpolicy.py ant16rooms
python experiments/SupervisedLearning.py ant16rooms 1000 500
python experiments/TestLTLspecs_Buchi.py ant16rooms 10

nohup python experiments/Generate_graph.py ant16rooms > training.log &