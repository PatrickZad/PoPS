python tools/train_ps_net.py --config-file configs/det_pretrain.yaml  --num-gpus 4 --resume --dist-url tcp://127.0.0.1:60888
python tools/train_ps_net.py --config-file configs/pops_cuhk.yaml  --num-gpus 4 --resume --dist-url tcp://127.0.0.1:60888
python tools/train_ps_net.py --config-file configs/pops_cuhk2prw.yaml  --num-gpus 4 --resume --dist-url tcp://127.0.0.1:60888
python tools/train_ps_net.py --config-file configs/pops_cuhk2prw2mvn.yaml  --num-gpus 4 --resume --dist-url tcp://127.0.0.1:60888