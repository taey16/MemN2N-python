#!/bin/sh
# for train
python demo/qa.py -d /storage/babi/tasks_1-20_v1-2/en -l /storage/babi/trained_model/en_PE_LS_RN_joint_hop4/ -m memn2n.pkl -seed 2 --train
python demo/qa.py -d /storage/babi/tasks_1-20_v1-2/en -l /storage/babi/trained_model/en_PE_LS_RN_joint_hop4/ -m memn2n.pkl -seed 2 --console
