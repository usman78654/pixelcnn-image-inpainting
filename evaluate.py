import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse

from model import PixelCNN

# Configuration
DATA_DIR = './data'
RESULTS_DIR = './results'
MODEL_PATH = './saved_models/pixelcnn_bedroom.pth'
IMAGE_SIZE = 64
OUTPUT_BINS = 256


def load_image(path, image_size=IMAGE_SIZE):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return transform(img)


def save_image(tensor, path):
    # tensor: C,H,W in [0,1]
    img = transforms.ToPILImage()(tensor.cpu())
    img.save(path)


def compute_mse(a, b):
    return float(((a - b) ** 2).mean())


def compute_psnr(mse, max_pixel=1.0):
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel) - 10 * np.log10(mse)


def quantize(img_tensor, bins=OUTPUT_BINS):
    # img_tensor in [0,1]
    return (img_tensor * (bins - 1)).round().long()


def dequantize(tensor, bins=OUTPUT_BINS):
    return tensor.float() / (bins - 1)


def reconstruct_argmax(model, occluded_tensor, device):
    # Run a single forward and take argmax per pixel/channel
    model.eval()
    with torch.no_grad():
        occluded = occluded_tensor.unsqueeze(0).to(device)  # 1,C,H,W
        out = model(occluded)  # (1, bins, C, H, W)
        probs = torch.softmax(out, dim=1)
        argmax = probs.argmax(dim=1)  # (1, C, H, W) with values in [0, bins-1]
        recon = dequantize(argmax.squeeze(0))  # C,H,W in [0,1]
    return recon


def main():
    parser = argparse.ArgumentParser(description='Evaluate PixelCNN image completion model')
    parser.add_argument('--dataset', type=str, default='val', 
                        choices=['val', 'train', 'test'],
                        help='Dataset to evaluate on: val (with ground truth), train (with ground truth), or test (no ground truth)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images to evaluate (useful for quick tests)')
    parser.add_argument('--save-results', action='store_true', default=True,
                        help='Save reconstructed images to results directory')
    args = parser.parse_args()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = PixelCNN(in_channels=3, n_filters=128, n_blocks=5, output_bins=OUTPUT_BINS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    print(f"Model loaded from {MODEL_PATH}\n")

    # Set up directories based on dataset
    if args.dataset == 'val':
        occ_dir = os.path.join(DATA_DIR, 'val', 'occluded_images')
        orig_dir = os.path.join(DATA_DIR, 'val', 'original_images')
        has_ground_truth = True
        print("Evaluating on VALIDATION set (with ground truth)")
    elif args.dataset == 'train':
        occ_dir = os.path.join(DATA_DIR, 'train', 'occluded_images')
        orig_dir = os.path.join(DATA_DIR, 'train', 'original_images')
        has_ground_truth = True
        print("Evaluating on TRAINING set (with ground truth)")
    else:  # test
        occ_dir = os.path.join(DATA_DIR, 'occluded_test')
        orig_dir = None
        has_ground_truth = False
        print("Evaluating on TEST set (no ground truth - visual inspection only)")

    if not os.path.exists(occ_dir):
        print(f'Error: Directory not found: {occ_dir}')
        return

    files = sorted([f for f in os.listdir(occ_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if len(files) == 0:
        print(f'No images found in {occ_dir}')
        return

    # Limit number of files if specified
    if args.limit:
        files = files[:args.limit]
        print(f"Limiting evaluation to {len(files)} images\n")
    else:
        print(f"Found {len(files)} images to evaluate\n")

    metrics = []
    for idx, fname in enumerate(files, 1):
        occ_path = os.path.join(occ_dir, fname)
        occluded = load_image(occ_path)

        # Load ground truth if available
        if has_ground_truth:
            # Derive original filename (remove 'occluded_' prefix)
            if fname.startswith('occluded_'):
                orig_name = fname.replace('occluded_', '', 1)
            else:
                orig_name = fname
            
            orig_path = os.path.join(orig_dir, orig_name)
            if not os.path.exists(orig_path):
                print(f'Warning: ground-truth original not found for {fname} (expected {orig_name}), skipping')
                continue
            original = load_image(orig_path)

        # Reconstruct image
        recon = reconstruct_argmax(model, occluded, device)

        # Save reconstruction
        if args.save_results:
            out_path = os.path.join(RESULTS_DIR, fname.replace('occluded_', 'recon_'))
            save_image(recon, out_path)

        # Compute metrics if ground truth available
        if has_ground_truth:
            mse = compute_mse(recon.numpy(), original.numpy())
            psnr = compute_psnr(mse)

            # Per-pixel exact match after quantization
            q_recon = quantize(recon)
            q_orig = quantize(original)
            exact_match = (q_recon == q_orig).float()
            per_pixel_acc = float(exact_match.mean())

            metrics.append({'file': fname, 'mse': mse, 'psnr': psnr, 'pixel_acc': per_pixel_acc})
            print(f"[{idx}/{len(files)}] {fname}: MSE={mse:.6f}, PSNR={psnr:.2f} dB, Pixel-Acc={per_pixel_acc:.4f}")
        else:
            print(f"[{idx}/{len(files)}] {fname}: Reconstruction saved (no ground truth)")

    # Print summary statistics
    if len(metrics) > 0:
        avg_mse = sum(m['mse'] for m in metrics) / len(metrics)
        avg_psnr = sum(m['psnr'] for m in metrics) / len(metrics)
        avg_acc = sum(m['pixel_acc'] for m in metrics) / len(metrics)
        
        print('\n' + '='*60)
        print('EVALUATION SUMMARY')
        print('='*60)
        print(f'Dataset: {args.dataset}')
        print(f'Images evaluated: {len(metrics)}')
        print(f'Average MSE: {avg_mse:.6f}')
        print(f'Average PSNR: {avg_psnr:.2f} dB')
        print(f'Average Pixel Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)')
        print('='*60)
        
        # Save metrics to file
        metrics_file = os.path.join(RESULTS_DIR, f'metrics_{args.dataset}.txt')
        with open(metrics_file, 'w') as f:
            f.write(f'Evaluation Results - {args.dataset} dataset\n')
            f.write('='*60 + '\n\n')
            f.write(f'Images evaluated: {len(metrics)}\n')
            f.write(f'Average MSE: {avg_mse:.6f}\n')
            f.write(f'Average PSNR: {avg_psnr:.2f} dB\n')
            f.write(f'Average Pixel Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)\n\n')
            f.write('Per-image metrics:\n')
            f.write('-'*60 + '\n')
            for m in metrics:
                f.write(f"{m['file']}: MSE={m['mse']:.6f}, PSNR={m['psnr']:.2f}, Acc={m['pixel_acc']:.4f}\n")
        print(f'\nMetrics saved to {metrics_file}')
    else:
        print('\nNo metrics computed (no ground truth available or no matching files)')
    
    if args.save_results:
        print(f'Reconstructed images saved to {RESULTS_DIR}/')



if __name__ == '__main__':
    main()
