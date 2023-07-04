from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
import os
import copy
import time
import tqdm

import torch
import pandas as pd
import clip.clip as clip
from clip.loss import ClipLoss
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize, FeatureDataset
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, get_logits, get_logits_noscale
from src.models.zeroshot import get_zeroshot_classifier
import src.datasets as datasets


def test_log_reg_warm_starting(train_features,
                               val_features,
                               test_features,
                               train_labels,
                               val_labels,
                               test_labels,
                               classification_head,
                               num_cs=100,
                               start_c=-1,
                               end_c=2,
                               max_iter=200,
                               random_state=0):
    Cs = np.logspace(start_c, end_c, num_cs)
    clf = LogisticRegression(random_state=random_state,
                             warm_start=True,
                             max_iter=max_iter)
    accs = []
    best_acc = -1.0
    best_clf, best_coef, best_intercept, best_i, best_c = None, None, None, None, None
    for i, C in zip(range(len(Cs)), Cs):
        clf.C = C
        clf.fit(train_features, train_labels)
        # val_acc = clf.score(val_features, val_labels)
        # test_acc = clf.score(test_features, test_labels)
        corrects = clf.predict(test_features) == test_labels
        wrong_ids = torch.arange(corrects.shape[0])[corrects == 0]
        wrong_ids_labels = test_labels[corrects == 0]
        print(wrong_ids)
        print(wrong_ids_labels)
        test_acc = corrects.mean()
        print(f"i : {i} c: {C} Val Acc : {0} Test Acc : {test_acc}")
        if test_acc > best_acc:
            best_acc = test_acc
            best_clf = copy.deepcopy(clf)
            best_coef = copy.deepcopy(clf.coef_)
            best_intercept = copy.deepcopy(clf.intercept_)
            best_i = i
            best_c = C

    return best_clf, best_coef, best_intercept, best_c, best_i, best_acc


def lbfgs(args, clip_encoder, classification_head, logger):
    assert args.train_dataset is not None, "Please provide a training dataset."
    logger.info('Linear Probe')
    model = clip_encoder
    input_key = 'images'
    preprocess_fn = clip_encoder.train_preprocess

    clip_encoder.process_images = True
    print_every = 100

    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(preprocess_fn,
                            location=args.data_location,
                            batch_size=args.batch_size)

    feature_dataset = FeatureDataset(is_train=True,
                                     image_encoder=model,
                                     dataset=dataset,
                                     device=args.device,
                                     cache_dir=args.cache_dir)

    num_batches = len(dataset.train_loader)
    print(f"Num Batches : {num_batches}")
    model = model.cuda()
    classification_head = classification_head.cuda()
    devices = list(range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids=devices)
    classification_head = torch.nn.DataParallel(classification_head,
                                                device_ids=devices)
    classification_head.train()
    model.train()

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    classifier_params = list(classification_head.parameters())
    total_params = classifier_params
    params = [p for p in total_params if p.requires_grad]

    # data_loader = get_dataloader(
    #         dataset, is_train=True, args=args, image_encoder=image_enc)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(clip_encoder.val_preprocess,
                                location=args.data_location,
                                batch_size=args.batch_size)
        feature_dataset = FeatureDataset(is_train=False,
                                         image_encoder=model,
                                         dataset=dataset,
                                         device=args.device,
                                         cache_dir=args.cache_dir)

    feat_path = os.path.join("/home/sachingo/im-lang-FT/scripts/",
                             args.cache_dir)
    train_feats_path = os.path.join(feat_path, args.train_dataset, "train")
    # val_feats_path = os.path.join(feat_path, args.eval_datasets[0], "val")
    test_feats_path = os.path.join(feat_path, args.eval_datasets[0], "val")
    # print(f"{train_feats_path},   {val_feats_path},   {test_feats_path}")
    train_features = torch.load(os.path.join(train_feats_path, "features.pt"))
    train_labels = torch.load(os.path.join(train_feats_path, "labels.pt"))
    # val_features = torch.load(os.path.join(val_feats_path, "features.pt"))
    # val_labels = torch.load(os.path.join(val_feats_path, "labels.pt"))
    test_features = torch.load(os.path.join(test_feats_path, "features.pt"))
    test_labels = torch.load(os.path.join(test_feats_path, "labels.pt"))

    best_clf, best_coef, best_intercept, best_c, best_i, best_acc = test_log_reg_warm_starting(
        train_features, None, test_features, train_labels, None, test_labels,
        classification_head)
    print(f"Best i {best_i} best c : {best_c} best acc : {best_acc}")
    logger.info(f"Best i {best_i} best c : {best_c} best acc : {best_acc}")
    torch.save(best_clf, os.path.join(feat_path, f"best_clf_{args.run}.pt"))
    torch.save(best_coef, os.path.join(feat_path, f"best_coef_{args.run}.pt"))
    torch.save(best_intercept,
               os.path.join(feat_path, f"best_intercept_{args.run}.pt"))
    torch.save(best_c, os.path.join(feat_path, f"best_c_{args.run}.pt"))
    torch.save(best_acc, os.path.join(feat_path, f"best_acc_{args.run}.pt"))
