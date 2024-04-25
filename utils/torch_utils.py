# Author: Harsh Kohli
# Date Created: 23-04-2024

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def pooling(outputs, inputs, strategy):
    if strategy == 'cls':
        outputs = outputs[:, 0]
    elif strategy == 'mean':
        outputs = torch.sum(outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
    else:
        raise NotImplementedError
    # outputs = F.normalize(outputs, p=2, dim=1)
    return outputs


def train_epoch(train_dataloader, accelerator, unet, text_encoder, noise_scheduler, args, optimizer, lr_scheduler,
                progress_bar):
    train_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet):
            encoder_hidden_states = text_encoder(batch["question_ids"]).last_hidden_state
            encoder_hidden_states = pooling(encoder_hidden_states, batch["question_ids"], 'cls')
            para_hidden_states = text_encoder(batch["para_ids"]).last_hidden_state
            para_hidden_states = pooling(para_hidden_states, batch["para_ids"], 'cls')

            latents = para_hidden_states.unsqueeze(1)
            latents = latents.unsqueeze(1)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)

            noise = torch.randn_like(latents)
            # noise = F.normalize(noise, p=2, dim=3)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item()

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)


def test_epoch(test_dataloader, unet, accelerator, noise_scheduler, text_encoder):
    total_correct_diff, total_correct_base, num_samples = 0, 0, 0
    for batch in tqdm(test_dataloader):
        encoder_hidden_states = text_encoder(batch["question_ids"].to(accelerator.device)).last_hidden_state
        encoder_hidden_states = pooling(encoder_hidden_states, batch["question_ids"], 'cls')
        para_hidden_states = text_encoder(batch["para_ids"].to(accelerator.device)).last_hidden_state
        para_hidden_states = pooling(para_hidden_states, batch["para_ids"], 'cls')

        pos_embeds = para_hidden_states

        neg_embeds = text_encoder(batch["neg_ids"].to(accelerator.device)).last_hidden_state
        neg_embeds = pooling(neg_embeds, batch["neg_ids"], 'cls')

        encoder_hidden_states = encoder_hidden_states.unsqueeze(1)

        para_hidden_states = para_hidden_states.unsqueeze(1)
        para_hidden_states = para_hidden_states.unsqueeze(1)
        latents = torch.randn_like(para_hidden_states)
        # latents = F.normalize(latents, p=2, dim=3)
        for t in noise_scheduler.timesteps:
            with torch.no_grad():
                noise_pred = unet(latents, t, encoder_hidden_states, return_dict=False)[0]

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            if t == 1:
                latents = noise_scheduler.step(noise_pred, t, latents).pred_original_sample
                break
            else:
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        model_pred = latents.squeeze()
        # model_pred = F.normalize(model_pred, p=2, dim=1)
        query_embeds = encoder_hidden_states.squeeze()

        pos_scores = F.cosine_similarity(model_pred, pos_embeds, dim=1)
        neg_scores = F.cosine_similarity(model_pred, neg_embeds, dim=1)
        more_similar = pos_scores > neg_scores
        num_correct = more_similar.sum().item()
        total_correct_diff = total_correct_diff + num_correct

        pos_base_scores = F.cosine_similarity(query_embeds, pos_embeds, dim=1)
        neg_base_scores = F.cosine_similarity(query_embeds, neg_embeds, dim=1)
        more_similar_base = pos_base_scores > neg_base_scores
        num_correct_base = more_similar_base.sum().item()
        total_correct_base = total_correct_base + num_correct_base

        bsz = query_embeds.shape[0]
        num_samples = num_samples + bsz

    return total_correct_diff, total_correct_base, num_samples
