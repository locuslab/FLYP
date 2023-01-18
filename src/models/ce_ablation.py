from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import os
import copy
import time
import tqdm

import torch
import pandas as pd
import clip.clip as clip
from clip.loss import ClipLoss

from src.args import parse_arguments
from torch.nn import functional as F
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, get_logits
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets.laion import get_data
import src.datasets as datasets


def ce_ablation(args, clip_encoder, classification_head, logger):
    assert args.train_dataset is not None, "Please provide a training dataset."
    logger.info('Fine-tuning Using FLYP Loss')
    model = clip_encoder
    input_key = 'images'
    preprocess_fn = clip_encoder.train_preprocess
    image_enc = None
    clip_encoder.process_images = True
    print_every = 100
    template = getattr(templates, args.template)

    dataset_class = getattr(datasets, args.train_dataset)
    print(f"Training dataset {args.train_dataset}")

    dataset = dataset_class(preprocess_fn,
                            location=args.data_location,
                            batch_size=args.batch_size)

    num_batches = len(dataset.train_loader)
    print(f"Num batches is {num_batches}")

    model = model.cuda()
    classification_head = classification_head.cuda()
    devices = list(range(torch.cuda.device_count()))
    logger.info('Using devices' + str(devices))
    model = torch.nn.DataParallel(model, device_ids=devices)
    classification_head = torch.nn.DataParallel(classification_head,
                                                device_ids=devices)
    classification_head.train()
    model.train()

    clip_params = list(model.parameters())
    total_params = clip_params
    params = [p for p in total_params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length,
                          args.epochs * num_batches, args.min_lr)

    ###Code for CE Ablation
    all_texts = []
    for classname in dataset.classnames:
        texts = []
        for t in template:
            texts.append(t(classname))
        texts = clip.tokenize(texts)  # tokenize
        all_texts.append(texts)

    all_texts = torch.stack(all_texts, dim=0)

    assert all_texts.shape[0] == len(dataset.classnames)
    assert all_texts.shape[1] == len(template)
    assert all_texts.shape[2] == 77
    #######

    stats = []
    for epoch in range(0, args.epochs):
        print("Epoch : ", epoch)
        epoch_stats = {}
        epoch_stats['epoch'] = epoch
        id_ce_loss_sum = 0
        model.train()
        model = model.cuda()
        classification_head.train()
        data_loader = get_dataloader(dataset,
                                     is_train=True,
                                     args=args,
                                     image_encoder=image_enc)

        for i, batch in enumerate(data_loader):
            start_time = time.time()
            step = i + epoch * num_batches
            if epoch != -1:
                scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch[input_key].cuda()
            labels = batch['labels'].cuda()

            #sample prompts for #C classes
            b = torch.arange(len(dataset.classnames))
            s = torch.randint(low=0,
                                high=all_texts.shape[1],
                                size=(all_texts.shape[0], ))
            current_texts = all_texts[b, s, :]
            current_texts = current_texts.cuda()
            assert current_texts.shape[0] == len(dataset.classnames)
            assert current_texts.shape[1] == 77

            ft_image_features = model(inputs, None)
            ft_text_features = model(None, current_texts)
            ft_image_features = ft_image_features / ft_image_features.norm(
                dim=-1, keepdim=True)
            ft_text_features = ft_text_features / ft_text_features.norm(
                dim=-1, keepdim=True)
            logit_scale = model.module.model.logit_scale.exp()

            assert ft_text_features.shape[0] == len(dataset.classnames)
            logits = logit_scale * ft_image_features @ ft_text_features.T
            xent_loss = F.cross_entropy(logits, labels)

            xent_loss.backward()
            optimizer.step()

            id_ce_loss_sum += xent_loss.item()

            if i % print_every == 0:
                percent_complete = 100 * i / num_batches
                logger.info(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"
                    f"ID FLYP Loss: {xent_loss.item():.4f}")

        id_ce_loss_avg = id_ce_loss_sum / num_batches

        # Evaluate
        args.current_epoch = epoch
        classification_head_new = get_zeroshot_classifier(
            args, model.module.model)
        classification_head_new = classification_head_new.cuda()

        eval_results = evaluate(model, args, classification_head_new,
                                epoch_stats, logger)

        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch}.pt')
            logger.info('Saving model to' + str(model_path))
            model.module.save(model_path)
            optim_path = os.path.join(args.save, f'optim_{epoch}.pt')
            torch.save(optimizer.state_dict(), optim_path)

        ood_acc = 0
        num_datasets = 0
        for k, v in epoch_stats.items():
            if 'Accuracy' in k:
                if k == 'ImageNet Accuracy':
                    #ignore the ID acc term
                    continue
                ood_acc += v
                num_datasets += 1
        if num_datasets != 0:
            ood_acc = ood_acc / num_datasets
        else:
            ood_acc = 0

        epoch_stats['Avg OOD Acc'] = round(ood_acc, 4)
        logger.info(f"Avg OOD Acc : {ood_acc:.4f}")
        logger.info(f"Avg ID FLYP Loss : {id_ce_loss_avg:.4f}")
        epoch_stats['Avg ID FLYP Loss'] = round(id_ce_loss_avg, 4)
        stats.append(epoch_stats)
        stats_df = pd.DataFrame(stats)
        log_dir = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
            args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
        os.makedirs(log_dir, exist_ok=True)
        stats_df.to_csv(log_dir + '/stats.tsv', sep='\t')

    if args.save is not None:
        return model_path