# Ultra-Lora
This fine-tuned SD model is trained with [kohya_ss LoRA](https://github.com/kohya-ss/sd-scripts) method.

## Installation
```bash
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade -r requirements.txt
pip install xformers==0.0.20

accelerate config
```
Answers to accelerate config:
```bash
- This machine
- No distributed training
- NO
- NO
- NO
- all
- fp16
```

## Fine-tuning SD with LoRA
![LoRA fine-tuning pipeline](/img/stable_diffusion_pipeline.png)

## Dataset
The common maternal fetal ultrasound planes [1]. The summary of this dataset.
| Category | No. Train | No. Test | No. Train (LoRA) |
| ----------- | ----------- | ----------- | ----------- |
| abdomen | 353  | 358  | 20 |
| brain   | 1620 | 1472 | 20 |
| femur   | 516  | 524  | 20 |
| thorax  | 1058 | 660  | 20 |
| cervix  | 981  | 645  | 20 |
| other   | 2601 | 1612 | 20 |

## LoRA Training Config
An example (**fetal abdomen**) of LoRA training configuration. 
```json
{
  "pretrained_model_name_or_path": "/root/autodl-tmp/sd_ckpt/v1-5-pruned.safetensors",
  "v2": false,
  "v_parameterization": false,
  "logging_dir": "/root/autodl-tmp/abdomen/datasets/log",
  "train_data_dir": "/root/autodl-tmp/abdomen/datasets/img",
  "reg_data_dir": "",
  "output_dir": "/root/autodl-tmp/abdomen/datasets/output",
  "max_resolution": "512,512",
  "learning_rate": "0.0001",
  "lr_scheduler": "constant",
  "lr_warmup": "0",
  "train_batch_size": 1,
  "epoch": "1",
  "save_every_n_epochs": "1",
  "mixed_precision": "fp16",
  "save_precision": "fp16",
  "seed": "1234",
  "num_cpu_threads_per_process": 2,
  "cache_latents": true,
  "caption_extension": ".txt",
  "enable_bucket": true,
  "gradient_checkpointing": false,
  "full_fp16": false,
  "no_token_padding": false,
  "stop_text_encoder_training": 0,
  "xformers": false,
  "save_model_as": "safetensors",
  "shuffle_caption": false,
  "save_state": true,
  "resume": "",
  "prior_loss_weight": 1.0,
  "text_encoder_lr": "5e-5",
  "unet_lr": "0.0001",
  "network_dim": 128,
  "lora_network_weights": "",
  "color_aug": false,
  "flip_aug": false,
  "clip_skip": 2,
  "gradient_accumulation_steps": 1.0,
  "mem_eff_attn": false,
  "output_name": "abdomen_v1.0",
  "model_list": "custom",
  "max_token_length": "75",
  "max_train_epochs": "1",
  "max_data_loader_n_workers": "1",
  "network_alpha": 128,
  "training_comment": "",
  "keep_tokens": "0",
  "lr_scheduler_num_cycles": "",
  "lr_scheduler_power": "",
  "persistent_data_loader_workers": false,
  "bucket_no_upscale": true,
  "random_crop": false,
  "bucket_reso_steps": 64.0,
  "caption_dropout_every_n_epochs": 0.0,
  "caption_dropout_rate": 0,
  "optimizer": "AdamW8bit",
  "optimizer_args": "",
  "noise_offset": "",
  "LoRA_type": "Standard",
  "conv_dim": 1,
  "conv_alpha": 1
}
```

## Inference

| LoRA STRENGTH | Sampler | Seed |
| ----------- | ----------- | ----------- |
| 0.5, 0.6, 0.7, 0.9, 1.0, 1.2, 1.5 | Euler a, Euler, DPM2, DPM2 a, DPM++ 2S a, DPM++ 2M, DPM++ SDE, DPM adaptive, DPM2 Karras, DPM2 a Karras, DPM++ SDE Karras, DPM++ 2M SDE Karras, DDIM | 3502338861 |

### Fetal Abdomen Example
```
# prompt
a photo of fetal abdomen <lora:abdomen_v1.0:STRENGTH>
```
![fetal abdomen](/img/00001-3614008528.png)

```
# prompt
a photo of fetal femur <lora:femur_v1.0:STRENGTH>
```
![fetal femur](/img/00002-3614008528.png)

```
# prompt
a photo of fetal brain <lora:brain_v1.0:STRENGTH>
```
![fetal brain](/img/0001-3502338861.png)

## Reference
[1] Burgos-Artizzu, X.P., Coronado-Gutiérrez, D., Valenzuela-Alcaraz, B. et al. Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes. Sci Rep 10, 10200 (2020). https://doi.org/10.1038/s41598-020-67076-5