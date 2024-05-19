# Author: Harsh Kohli
# Date Created: 23-04-2024

import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from utils.data_utils import write_logs


class DiffusionMLP(nn.Module):
    def __init__(self,
                 n_timesteps,
                 d_embedding,
                 d_in,
                 d_hidden
                 ):
        super().__init__()

        self.time_embedding = nn.Embedding(n_timesteps, d_embedding)

        self.layers = nn.Sequential(
            nn.Linear(d_in * 2 + d_embedding, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_in),
        )

    def forward(self, query_embed, noisy_embed, timesteps):
        timesteps = self.time_embedding(timesteps)
        x = torch.cat([query_embed, noisy_embed, timesteps], -1)
        noise_prediction = self.layers(x)
        return noise_prediction


def pooling(outputs, inputs, strategy, normalize):
    if strategy == 'cls':
        outputs = outputs[:, 0]
    elif strategy == 'mean':
        outputs = torch.sum(outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
    else:
        raise NotImplementedError
    if normalize:
        outputs = F.normalize(outputs, p=2, dim=1)
    return outputs


def train_epoch(train_dataloader, accelerator, model, text_encoder, noise_scheduler, args, optimizer, lr_scheduler,
                progress_bar):
    train_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            query_embeddings = text_encoder(batch["question_ids"]).last_hidden_state
            query_embeddings = pooling(query_embeddings, batch["question_ids"], 'cls', args.normalize)
            pos_para_embeddings = text_encoder(batch["para_ids"]).last_hidden_state
            pos_para_embeddings = pooling(pos_para_embeddings, batch["para_ids"], 'cls', args.normalize)
            # neg_para_embeddings = text_encoder(batch["neg_ids"]).last_hidden_state
            # neg_para_embeddings = pooling(neg_para_embeddings, batch["neg_ids"], 'cls', args.normalize)
            bsz = query_embeddings.shape[0]

            latents = pos_para_embeddings
            if args.model_type == 'unet':
                latents = pos_para_embeddings.unsqueeze(1)
                latents = latents.unsqueeze(1)
                query_embeddings = query_embeddings.unsqueeze(1)
            noise = torch.randn_like(latents)

            if args.normalize:
                noise = F.normalize(noise, p=2, dim=3)

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            if args.model_type == 'unet':
                model_pred = model(noisy_latents, timesteps, query_embeddings, return_dict=False)[0]
            else:
                model_pred = model.forward(query_embeddings, noisy_latents, timesteps)

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item()

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            break


def test_epoch(test_dataloader, model, accelerator, noise_scheduler, text_encoder, normalize, log_file, logger,
               model_type):
    total_correct_diff, total_correct_base, num_samples = 0, 0, 0
    dist_query_to_pred, dist_query_to_pos, dist_query_to_neg, dist_pred_to_pos, dist_pred_to_neg = [], [], [], [], []
    sim_pos, sim_neg, sim_pos_base, sim_neg_base = [], [], [], []
    for batch in tqdm(test_dataloader):
        query_embeddings = text_encoder(batch["question_ids"].to(accelerator.device)).last_hidden_state
        query_embeddings = pooling(query_embeddings, batch["question_ids"], 'cls', normalize)
        pos_para_embeddings = text_encoder(batch["para_ids"].to(accelerator.device)).last_hidden_state
        pos_para_embeddings = pooling(pos_para_embeddings, batch["para_ids"], 'cls', normalize)
        neg_para_embeddings = text_encoder(batch["neg_ids"].to(accelerator.device)).last_hidden_state
        neg_para_embeddings = pooling(neg_para_embeddings, batch["neg_ids"], 'cls', normalize)
        bsz = query_embeddings.shape[0]

        if model_type == 'unet':
            query_embeddings = query_embeddings.unsqueeze(1)
            pos_para_embeddings = pos_para_embeddings.unsqueeze(1)
            pos_para_embeddings = pos_para_embeddings.unsqueeze(1)

        latents = torch.randn_like(pos_para_embeddings)
        if normalize:
            latents = F.normalize(latents, p=2, dim=3)

        for t in noise_scheduler.timesteps:
            with torch.no_grad():
                if model_type == 'unet':
                    noise_pred = model(latents, t, query_embeddings, return_dict=False)[0]
                else:
                    timesteps = torch.zeros(bsz, device=query_embeddings.device).long() + t
                    noise_pred = model.forward(query_embeddings, latents, timesteps)

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        model_pred = latents.squeeze()

        if normalize:
            model_pred = F.normalize(model_pred, p=2, dim=1)
        query_embeddings = query_embeddings.squeeze()
        pos_para_embeddings = pos_para_embeddings.squeeze()

        pos_scores = F.cosine_similarity(model_pred, pos_para_embeddings, dim=1)
        neg_scores = F.cosine_similarity(model_pred, neg_para_embeddings, dim=1)
        more_similar = pos_scores > neg_scores
        num_correct = more_similar.sum().item()
        total_correct_diff = total_correct_diff + num_correct

        pos_base_scores = F.cosine_similarity(query_embeddings, pos_para_embeddings, dim=1)
        neg_base_scores = F.cosine_similarity(query_embeddings, neg_para_embeddings, dim=1)
        more_similar_base = pos_base_scores > neg_base_scores
        num_correct_base = more_similar_base.sum().item()
        total_correct_base = total_correct_base + num_correct_base

        num_samples = num_samples + bsz

        dist_query_to_pred.append(torch.mean((query_embeddings - model_pred) ** 2).item())
        dist_query_to_pos.append(torch.mean((query_embeddings - pos_para_embeddings) ** 2).item())
        dist_query_to_neg.append(torch.mean((query_embeddings - neg_para_embeddings) ** 2).item())
        dist_pred_to_pos.append(torch.mean((model_pred - pos_para_embeddings) ** 2).item())
        dist_pred_to_neg.append(torch.mean((model_pred - neg_para_embeddings) ** 2).item())
        sim_pos.append(torch.mean(pos_scores).item())
        sim_neg.append(torch.mean(neg_scores).item())
        sim_pos_base.append(torch.mean(pos_base_scores).item())
        sim_neg_base.append(torch.mean(neg_base_scores).item())
        break

    write_logs(total_correct_diff, total_correct_base, num_samples, dist_query_to_pred, dist_query_to_pos,
               dist_query_to_neg, dist_pred_to_pos, dist_pred_to_neg, sim_pos, sim_neg, sim_pos_base, sim_neg_base,
               log_file, logger)
