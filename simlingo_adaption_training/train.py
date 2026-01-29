import os
import hydra

from omegaconf import OmegaConf
import torch
import wandb


from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ThroughputMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from transformers import AutoProcessor

from pathlib import Path
from simlingo_adaption_training.utils.logging_project import setup_logging, sync_wandb
from simlingo_adaption_training.config import TrainConfig
from simlingo_adaption_training.callbacks.visualise import VisualiseCallback

def check_gradient(cfg,model):
    # 注册 Hook 以检查 Scheduler 梯度
    if cfg.adaption_train:
        def print_grad(name):
            def hook(grad):
                if grad is not None:
                    grad_norm = grad.norm().item()
                    print(f"[Gradient Hook] {name} grad norm: {grad_norm}, shape: {grad.shape}")
                else:
                    print(f"[Gradient Hook] {name} has None gradient")
            return hook

        # 我们需要访问 model 内部的 scheduler 实例
        # 注意：model 是 DrivingModel，它内部持有 language_model，scheduler 可能在 language_model 或 DrivingModel 中
        # 根据 driving.py，scheduler 是 DrivingModel 的成员 self.scheduler
        if hasattr(model, 'scheduler'):
            # 注册给 scheduler 的 MLP head 输出权重
            if hasattr(model.scheduler, 'mlp_head'):
                model.scheduler.mlp_head.weight.register_hook(print_grad("Scheduler MLP Head Weight"))
            
            # 注册给 scheduler_up_proj
            if hasattr(model.scheduler, 'scheduler_up_proj'):
                 # 假设 scheduler_up_proj 是 FeedForward，取第二层线性层检查
                 if hasattr(model.scheduler.scheduler_up_proj, 'net'):
                     model.scheduler.scheduler_up_proj.net[-1].weight.register_hook(print_grad("Scheduler UpProj Last Layer Weight"))

        # [新增] 检查 LLM 各层是否参与训练 (根据 num_prefix_layers)
        # 使用更健壮的 named_parameters 遍历方法，不依赖具体的模型嵌套结构
        print("[Gradient Hook Setup] Scanning language model for trainable parameters...")
        lm = model.language_model
        
        monitored_layers = set()
        
        # 遍历所有参数，寻找可训练的参数
        for name, param in lm.named_parameters():
            if param.requires_grad:
                # name 示例: model.layers.0.self_attn.q_proj.lora_A.default.weight
                if "layers" in name and "lora" in name: 
                    import re
                    # 提取层索引
                    match = re.search(r"layers\.(\d+)\.", name)
                    if match:
                        layer_idx = int(match.group(1))
                        # 为了避免日志刷屏，我们只监听 q_proj 的 lora_B (它直接影响输出)
                        if "q_proj" in name and "lora_B" in name:
                            monitored_layers.add(layer_idx)
                            print(f"[Gradient Hook Setup] Hooking LLM Layer {layer_idx}: {name}")
                            param.register_hook(print_grad(f"LLM L{layer_idx} {name.split('.')[-2]}")) # print "lora_B"
        
        if monitored_layers:
            print(f"[Gradient Hook Setup] Monitored LLM Layers (q_proj LoRA): {sorted(list(monitored_layers))}")
        else:
            print("[Gradient Hook Setup] No trainable LLM layers found for monitoring.")

@hydra.main(config_path=f"config", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(cfg.seed, workers=True)

    # turn off wandb uploading when in debug mode
    if cfg.debug:
        os.environ["WANDB_MODE"] = "offline"
    
    cfg.wandb_name = f"{cfg.wandb_name}_{cfg.name}"
    
    processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)
    model_type_name = cfg.model.vision_model.variant.split('/')[1]
    cache_dir = None #f"pretrained/{(model_type_name)}"
    # hydra根据配置文件中的 _target_ 字段，动态导入并创建类实例。
    data_module = hydra.utils.instantiate(
        cfg.data_module, 
        processor=processor,
        encoder_variant=cfg.model.vision_model.variant,
        llm_variant=cfg.model.language_model.variant,
        _recursive_=False
    )
    
    if cfg.adaption_train:
        cfg.model.adaption_train = True
        cfg.model.simlingo_checkpoint = cfg.simlingo_checkpoint
        cfg.model.vision_model.freeze = True
    
        cfg.model.language_model.adaption_train = True # adaption,需要加载simlingo预训练权重
        cfg.model.language_model.num_prefix_layers = cfg.model.scheduler_model.num_prefix_layers# align
    
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=cache_dir,
        _recursive_=False
        )
        
    
    # 是否加载checkpoint
    if cfg.checkpoint is not None:
        current_path = Path(__file__).resolve()
        project_path = current_path.parent.parent
        checkpoint = os.path.join(project_path, cfg.checkpoint)
        if os.path.isdir(checkpoint): # 文件夹则合并和转换DeepSpeed的checkpoints
            state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint)
        else:
            state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict) # 加载checkpoint应该严格匹配
        

        
    # print config
    print(OmegaConf.to_yaml(cfg))
    os.environ["WANDB_DISABLE_CODE"] = "True"
    
    ## check gradient
    #check_gradient(cfg,model)


    if cfg.overfit > 0:
        overfit = cfg.overfit
        
    # setup logging
    setup_logging(cfg)

    # resume training
    resume_path = cfg.resume_path
    resume_wandb = False

    # if folder for this experiment does not exist set resume to true
    # to create necessary folders to resume wandb logging later
    if resume_path is not None and not os.path.exists(resume_path):
        resume_wandb = True
    elif resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        resume_wandb = True

    if resume_path is not None and os.path.exists(resume_path) and cfg.resume:
        resume_path = resume_path
    else:
        resume_path = None

    # setup lightning logger
    loggers = []
    # csvlogger = CSVLogger("log/", "CSVLogger")
    # loggers.append(csvlogger)
    # csvlogger = None

    wandblogger = WandbLogger(
        project=cfg.wandb_project,
        id=cfg.wandb_name,
        name=cfg.wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        resume=resume_wandb,
    )
    wandblogger.watch(model)
    loggers.append(wandblogger)

    # 多卡训练策略
    strategy = cfg.strategy
    if strategy == "deepspeed_stage_2":
        strategy = pl.strategies.DeepSpeedStrategy(
            stage=2, loss_scale=cfg.fp16_loss_scale, logging_batch_size_per_gpu=cfg.data_module.batch_size
        )

    # 模型权重保存回调函数
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1, # 保留每个epoch的模型权重
        monitor="val/loss",
        mode="min",
        dirpath="./checkpoints",
        filename="{epoch:03d}",
        save_last=True,
        every_n_epochs=cfg.val_every_n_epochs,
        # every_n_train_steps=cfg.val_check_interval,
    )

    # 学习率变化
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_summary = ModelSummary(max_depth=3)
    callbacks=[
        checkpoint_callback, 
        model_summary, 
        # ThroughputMonitor(batch_size_fn=lambda batch: batch.driving_input.camera_images.size(0)), 
        #VisualiseCallback(interval=1000, val_interval=1000) # 每隔interval可视化预测结果， 可视化路径点对比图和文本预测对比图
        VisualiseCallback(interval=1, val_interval=1) # 每隔一定步数可视化预测结果， 可视化路径点对比图和文本预测对比图
    ]
    if not cfg.debug: 
        callbacks.append(lr_monitor)
    
    print(f"Number of GPUS: {cfg.gpus}")
    overfit = 0
    
    if cfg.gpus >= 1:
        trainer = Trainer(
            accelerator="gpu",
            benchmark=True,
            callbacks=callbacks,
            devices=cfg.gpus,
            # enable_checkpointing=False,
            num_sanity_val_steps=4,
            gradient_clip_val=0.3,
            # gradient_clip_algorithm="value",
            # log_every_n_steps=10,
            logger=loggers,
            # max_steps=cfg.max_steps,
            precision=cfg.precision,
            strategy=strategy,
            sync_batchnorm=True,
            # use_distributed_sampler=False,
            max_epochs=cfg.max_epochs,
            overfit_batches=overfit,
            check_val_every_n_epoch=cfg.val_every_n_epochs,
            # val_check_interval=cfg.val_check_interval,
        )

    trainer.fit(model, data_module, ckpt_path=resume_path)
    wandb.finish()

if __name__ == "__main__":
    main()