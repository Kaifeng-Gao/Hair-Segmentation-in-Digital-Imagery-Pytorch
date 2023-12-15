from tqdm import tqdm
import utils
import os
import argparse

from datasets import FigaroDataset
from metrics import StreamSegMetrics
from torchvision import transforms as T

import torch
import torch.nn as nn
import numpy as np

from PIL import Image, ImageOps
from glob import glob
import segmentation_models_pytorch as smp


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='hair',
                        choices=['hair'], help='Name of training set')

    # Model Options
    parser.add_argument("--model", type=str, default='DeepLabV3Plus',
                        choices=["Unet", "UnetPlusPlus", "FPN", "PAN", "PSPNet", "DeepLabV3", "DeepLabV3Plus"], help='model name')
    parser.add_argument("--encoder", type=str, default='resnet101', choices=['mobilenet_v2', 'resnet101', 'resnet50'],
                        help="encoder name")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def pad_image(img, target_height, target_width):
    pad_height = target_height - img.size[1]
    pad_width = target_width - img.size[0]

    padding = (pad_width // 2, pad_height // 2, pad_width - (pad_width // 2), pad_height - (pad_height // 2))
    padded_img = ImageOps.expand(img, border=padding, fill='white')

    return padded_img

def main():
    opts = get_argparser().parse_args()
    opts.num_classes = 2
    decode_fn = FigaroDataset.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s' % (ext)), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)

    # Set up model
    model_class = getattr(smp, opts.model)
    model = model_class(
        encoder_name=opts.encoder,
        in_channels=3,
        classes=opts.num_classes,
    )
    utils.set_bn_momentum(model.encoder, momentum=0.01)
    metrics = StreamSegMetrics(opts.num_classes)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("Checkpoint not found")
        exit(1)

    if opts.crop_val:
        transform = T.Compose([
            T.Resize(opts.crop_size),
            T.CenterCrop(opts.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext) - 1]
            img = Image.open(img_path).convert('RGB')

            target_height = int(np.ceil(img.size[1] / 16) * 16)
            target_width = int(np.ceil(img.size[0] / 16) * 16)
            img = pad_image(img, target_height, target_width)

            img = transform(img).unsqueeze(0)  # To tensor of NCHW
            img = img.to(device)
            pred = model(img).max(1)[1].cpu().numpy()[0]  # HW

            label_path = img_path.replace("Original", "GT").replace("-org.jpg", "-gt.pbm")
            label = pad_image(Image.open(label_path), target_height, target_width)
            targets = np.array(label)

            metrics.update(targets, pred)
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            if opts.save_val_results_to:
                colorized_preds.save(os.path.join(opts.save_val_results_to, img_name + '.png'))
        metrics_message = metrics.to_str(metrics.get_results())
        tqdm.write(metrics_message)

if __name__ == '__main__':
    main()
