save_path="xxx"
nproc_per_node=8

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=xxx \
    data.val_files=xxx \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['prompt'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=x \
    model.partial_pretrain=xxx \
    trainer.default_local_dir=$save_path \
    trainer.project_name=xxx \
    trainer.experiment_name=xxx \
    trainer.total_epochs=3 \
    trainer.logger=['console','tensorboard'] \
    trainer.default_hdfs_dir=null \
