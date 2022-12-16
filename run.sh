python train_protonet.py --innerlr 1e-3 --metalr 1e-3 --gamma 0.1 --dataset MiniImageNet --dim 640 --curvaturedim 640 \
--max_epoch 60 --step_size 10 --query_num 30 --temperature 1 --validation_way 5 --train_way 10 \
--shot 1 --train_step 5 --sample_num 15  --augment_lambda 2 --tradeoff1 1e-5 --tradeoff2 2 --curvature 0 --curvaturel 0.1  --curvaturescale -0.3  --curvaturestart -0.1 >& mini_shot1_epoch60.log


python train_protonet.py --innerlr 1e-3 --metalr 5e-3 --gamma 0.1 --dataset MiniImageNet --dim 640 --curvaturedim 640 \
--max_epoch 60 --step_size 10 --query_num 30 --temperature 1 --validation_way 5 --train_way 10 \
--shot 5 --train_step 5 --sample_num 15  --augment_lambda 2 --tradeoff1 1e-4 --tradeoff2 2 --curvature 0 --curvaturel 0.1  --curvaturescale -0.3  --curvaturestart -0.7 >& mini_shot5_epoch60.log


python train_protonet.py --innerlr 1e-3 --metalr 1e-3 --gamma 0.1 --dataset CUB --dim 640 --curvaturedim 640 \
--max_epoch 40 --step_size 10 --query_num 30 --temperature 1 --validation_way 5 --train_way 10 \
--shot 1 --train_step 10 --sample_num 15  --augment_lambda 2 --tradeoff1 1e-4 --tradeoff2 2 --curvature 0 --curvaturel 0.1  --curvaturescale -0.3  --curvaturestart -0.7 >& cub_shot1.log


python train_protonet.py --innerlr 1e-3 --metalr 1e-3 --gamma 0.1 --dataset CUB --dim 640 --curvaturedim 640 \
--max_epoch 40 --step_size 10 --query_num 30 --temperature 1 --validation_way 5 --train_way 10 \
--shot 5 --train_step 5 --sample_num 15  --augment_lambda 2 --tradeoff1 1e-4 --tradeoff2 2 --curvature 0 --curvaturel 0.1  --curvaturescale -0.3  --curvaturestart -0.7 >& cub_shot5.log
