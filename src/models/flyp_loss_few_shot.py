import torch
from clip.loss import ClipLoss

from src.datasets.common import get_dataloader
from src.models.eval import eval_single_batch_dataset, eval_single_dataset
from src.models.utils import cosine_lr
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets.laion import get_data
import src.datasets as datasets

from copy import deepcopy


def flyp_loss_few_shot(args, clip_encoder, classification_head, logger):
    assert args.train_dataset is not None, "Please provide a training dataset."

    model = clip_encoder

    clip_encoder.process_images = True

    give_batch_size = args.batch_size
    args.batch_size = args.k
    num_batches = 1

    img_text_data = get_data(
        args, (clip_encoder.train_preprocess, clip_encoder.val_preprocess),
        epoch=0)
    assert len(
        img_text_data), 'At least one train or eval dataset must be specified.'
    ft_dataloader = img_text_data['train_ft'].dataloader
    ft_iterator = iter(ft_dataloader)
    args.batch_size = give_batch_size

    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    clip_loss_fn = ClipLoss(local_loss=False,
                            gather_with_grad=False,
                            cache_labels=True,
                            rank=0,
                            world_size=1,
                            use_horovod=False)

    clip_params = list(model.parameters())
    total_params = clip_params
    params = [p for p in total_params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length,
                          args.epochs * num_batches, args.min_lr)

    images0 = []
    texts0 = []
    images1 = []
    texts1 = []
    match = None
    while True:
        ft_batch = next(ft_iterator)
        ft_image, ft_text = ft_batch
        if match is None:
            match = ft_text[0, :]

        for i in range(ft_text.shape[0]):
            if torch.equal(match, ft_text[i]):
                if len(texts0) < args.k:
                    texts0.append(ft_text[i])
                    images0.append(ft_image[i])
            else:
                if len(texts1) < args.k:
                    texts1.append(ft_text[i])
                    images1.append(ft_image[i])

        if len(texts0) == args.k and len(texts1) == args.k:
            break

    ft_image = torch.stack(images0 + images1, dim=0)
    ft_text = torch.stack(texts0 + texts1, dim=0)

    # get val data
    val_dataset_name = None
    for i, dataset_name in enumerate(args.eval_datasets):
        if 'Val' in dataset_name:
            val_dataset_name = dataset_name
            break
    assert val_dataset_name is not None, 'please give val data'
    print('Evaluating on', val_dataset_name)
    val_dataset_class = getattr(datasets, val_dataset_name)
    val_dataset = val_dataset_class(model.module.val_preprocess,
                                    location=args.data_location,
                                    batch_size=args.k)
    val_dataloader = get_dataloader(val_dataset,
                                    is_train=False,
                                    args=args,
                                    image_encoder=None)
    val_iterator = iter(val_dataloader)

    images0 = []
    texts0 = []
    images1 = []
    texts1 = []
    match = None
    while True:
        val_batch = next(val_iterator)
        img, txt = val_batch[0], val_batch[1]
        if match is None:
            match = txt[0]

        for i in range(img.shape[0]):
            if match == txt[i]:
                if len(texts0) < args.k:
                    texts0.append(txt[i])
                    images0.append(img[i])
            else:
                if len(texts1) < args.k:
                    texts1.append(txt[i])
                    images1.append(img[i])

        if len(texts0) == args.k and len(texts1) == args.k:
            break

    img = torch.stack(images0 + images1, dim=0)
    txt = torch.tensor(texts0 + texts1, dtype=torch.long)

    val_batch = [img, txt]

    max_val = 0
    min_cnt_loss = 1e10

    val_dataset = val_dataset_class(model.module.val_preprocess,
                                    location=args.data_location,
                                    batch_size=args.batch_size)
    model_copy = None
    for epoch in range(-1, args.epochs):
        print("Epoch : ", epoch)
        epoch_stats = {}
        epoch_stats['epoch'] = epoch
        id_contrastive_loss_sum = 0
        model.train()
        model = model.cuda()

        if epoch != -1:
            for i in range(num_batches):
                step = i + epoch * num_batches
                if epoch != -1:
                    scheduler(step)
                optimizer.zero_grad(set_to_none=True)

                assert ft_image.shape[0] == 2 * args.k, 'batch mismatch'
                ft_image, ft_text = ft_image.cuda(), ft_text.cuda()

                ft_image_features, ft_text_features, logit_scale2 = model(
                    ft_image, ft_text)

                try:
                    ls = logit_scale2[0]
                except Exception:
                    ls = logit_scale2
                ft_clip_loss = clip_loss_fn(ft_image_features,
                                            ft_text_features, ls)
                if epoch != -1:
                    ft_clip_loss.backward(retain_graph=False)

                id_contrastive_loss_sum += ft_clip_loss.item()

                if epoch != -1:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                ft_image, ft_text = ft_image.cpu(), ft_text.cpu()

        with torch.no_grad():
            # Evaluate
            args.current_epoch = epoch
            classification_head_new = get_zeroshot_classifier(
                args, model.module.model)
            classification_head_new = classification_head_new.cuda()
            val_acc, cnt_loss = eval_single_batch_dataset(
                model, val_dataset, args, classification_head_new, val_batch)

            # print(val_acc)

            logger.info(f"Epoch {epoch} results {val_acc}")

            if cnt_loss <= min_cnt_loss:
                max_val = val_acc
                min_cnt_loss = cnt_loss

                model_copy = deepcopy(model.cpu())

                for param in model_copy.parameters():
                    param.requires_grad = False

    del ft_image
    del ft_text

    model = model_copy
    model = model.cuda()
    classification_head_new = get_zeroshot_classifier(args, model.module.model)
    classification_head_new = classification_head_new.cuda()
    val_acc, cnt_loss = eval_single_batch_dataset(model, val_dataset, args,
                                                  classification_head_new,
                                                  val_batch)

    assert val_acc == max_val, f'max val not matching Max {max_val}, new {val_acc}'
    assert cnt_loss == min_cnt_loss, f'min val not matching Max {min_cnt_loss}, new {cnt_loss}'

    test_dataset_name = None
    for i, dataset_name in enumerate(args.eval_datasets):
        if 'Test' in dataset_name:
            test_dataset_name = dataset_name
            break
    assert test_dataset_name is not None, 'please give test data'
    print('Evaluating on', test_dataset_name)
    test_dataset_class = getattr(datasets, test_dataset_name)
    test_dataset = test_dataset_class(model.module.val_preprocess,
                                      location=args.data_location,
                                      batch_size=args.batch_size)

    results = eval_single_dataset(model, test_dataset, args,
                                  classification_head_new)

    test_acc = round(results['top1'], 4)

    return val_acc, test_acc