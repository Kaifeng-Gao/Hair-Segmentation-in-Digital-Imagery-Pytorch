from tqdm import tqdm
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import FigaroDataset
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import torchvision.transforms as std_trnsf
from utils import joint_transforms as jnt_trnsf
import segmentation_models_pytorch as smp


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='hair',
                        choices=['hair'], help='Name of dataset')

    # Model Options
    parser.add_argument("--model", type=str, default='DeepLabV3Plus',
                        choices=["Unet", "UnetPlusPlus", "FPN", "PAN", "PSPNet", "DeepLabV3", "DeepLabV3Plus"], help='model name')
    parser.add_argument("--encoder", type=str, default='resnet101', choices=['mobilenet_v2', 'resnet101', 'resnet50'],
                        help="encoder name")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=500,
                        help="epoch number (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=3,
                        help='batch size for validation (default: 3)')
    parser.add_argument("--crop_size", type=int, default=256)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 10)")
    parser.add_argument("--download", action='store_true', default=True,
                        help="download datasets")

    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_joint_transforms = jnt_trnsf.Compose([
        jnt_trnsf.RandomCrop(opts.crop_size),
        jnt_trnsf.RandomRotate(5),
        jnt_trnsf.RandomHorizontallyFlip()
    ])

    # transforms only on images
    train_image_transforms = std_trnsf.Compose([
        std_trnsf.ColorJitter(0.05, 0.05, 0.05, 0.05),
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    test_joint_transforms = jnt_trnsf.Compose([
        jnt_trnsf.RandomCrop(opts.crop_size),
    ])

    test_image_transforms = std_trnsf.Compose([
        std_trnsf.ToTensor(),
        std_trnsf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dst = FigaroDataset(root_dir=opts.data_root,
                                joint_transforms=train_joint_transforms,
                                image_transforms=train_image_transforms,
                                mask_transforms=None,
                                download=opts.download,
                                train=True)
    val_dst = FigaroDataset(root_dir=opts.data_root,
                            joint_transforms=test_joint_transforms,
                            image_transforms=test_image_transforms,
                            mask_transforms=None,
                            download=False,
                            train=False)
    return train_dst, val_dst


def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score


def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 8

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    

    # Setup dataloader
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model_class = getattr(smp, opts.model)
    model = model_class(
        encoder_name=opts.encoder,
        in_channels=3,
        classes=8,
    )
    utils.set_bn_momentum(model.encoder, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.encoder.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.segmentation_head.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    pbar = tqdm(total=opts.total_itrs, desc='Loss: N/A, Val Mean IoU: N/A')
    while True:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            pbar.update(1)
            pbar.set_description(f'Epoch {cur_epochs}, Itrs {cur_itrs}/{opts.total_itrs}, Loss={np_loss:.4f}')

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_%s_os%d.pth' %
                          (opts.model, opts.encoder, opts.dataset, opts.output_stride))
                model.eval()
                val_score = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)
                metrics_message = metrics.to_str(val_score)
                tqdm.write(metrics_message)
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_%s_os%d.pth' %
                              (opts.model, opts.encoder, opts.dataset, opts.output_stride))
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()
