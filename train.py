# Author: Harsh Kohli
# Date Created: 22-04-2024

import os
import logging
import argparse
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from transformers import AutoModel, AutoTokenizer
from diffusers.optimization import get_scheduler
from diffusers import UNet2DConditionModel, DDPMScheduler
from utils.data_utils import load_all_datasets, preprocess_dataset, preprocess_test_dataset, collate_fn
from utils.torch_utils import test_epoch, train_epoch, DiffusionMLP

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Training diffusion models for text embedding.")

    parser.add_argument("--embedding_model_path", type=str, default='Alibaba-NLP/gte-base-en-v1.5',
                        help="Base embedding model for query/passage.")
    parser.add_argument("--model_type", type=str, default="unet", choices=["mlp", "unet"],
                        help="Model type for noise prediction.")
    parser.add_argument("--embedding_size", type=int, default=768, help="Embedding dimensions of the text encoder.")
    parser.add_argument("--time_embedding_size", type=int, default=32,
                        help="Time embedding dimensions for custom models.")

    parser.add_argument("--max_tokens", type=int, default=512, help="Embedding dimensions of the text encoder.")
    parser.add_argument("--log_file", type=str, default='logs/logs8.txt', help="Log file.")

    parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--train_epochs", type=int, default=500, help="Total training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after warmup).")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--num_train_timesteps", type=int, default=1000,
                        help="Number of denoising steps during training.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps during inference.")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup",
                        choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant',
                                 'constant_with_warmup'], help='The learning rate scheduler type to use.')
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps in the lr scheduler.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--normalize", default=False, type=bool, help="Normalize embeddings everywhere.")

    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for logs and checkpoints.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="TensorBoard log directory.")
    parser.add_argument("--data_dir", type=str, default="dataset_small", help="Train Datasets.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
                        help="Mixed precision type. Select 'no' if not needed.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(mixed_precision=args.mixed_precision, log_with="tensorboard",
                              project_config=accelerator_project_config)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)

    logger.info(accelerator.state)
    os.makedirs(args.output_dir, exist_ok=True)
    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                    num_train_timesteps=args.num_train_timesteps,
                                                    subfolder="scheduler")
    noise_scheduler.set_timesteps(args.num_inference_steps)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path)
    text_encoder = AutoModel.from_pretrained(args.embedding_model_path, trust_remote_code=True).cuda()

    if args.model_type == 'unet':
        model = UNet2DConditionModel(
            in_channels=1, out_channels=1, cross_attention_dim=args.embedding_size,
            block_out_channels=[32, 64, 128, 256]
        )
    else:
        model = DiffusionMLP(args.num_train_timesteps, args.time_embedding_size, args.embedding_size,
                             args.embedding_size)

    text_encoder.requires_grad_(False)
    model.train()
    torch.backends.cuda.matmul.allow_tf32 = True
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    all_datasets = load_all_datasets(args.data_dir)
    train_dataset = all_datasets["train"].with_transform(lambda x: preprocess_dataset(x, tokenizer, args.max_tokens))
    test_dataset = all_datasets["test"].with_transform(lambda x: preprocess_test_dataset(x, tokenizer, args.max_tokens))

    train_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
    )

    iterations_per_epoch = len(train_dataloader)
    max_train_steps = args.train_epochs * iterations_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader,
                                                                          lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    text_encoder.to(accelerator.device, dtype=weight_dtype)

    iterations_per_epoch = len(train_dataloader)
    max_train_steps = args.train_epochs * iterations_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    initial_global_step = 0
    progress_bar = tqdm(range(0, max_train_steps), initial=initial_global_step, desc="Steps")
    for epoch in range(args.train_epochs):
        train_epoch(train_dataloader, accelerator, unet, text_encoder, noise_scheduler, args, optimizer, lr_scheduler,
                    progress_bar)
        logger.info("***** Running validation *****")
        test_epoch(test_dataloader, unet, accelerator, noise_scheduler, text_encoder, args.normalize, args.log_file,
                   logger, args.model_type)
    accelerator.end_training()


if __name__ == "__main__":
    main()
