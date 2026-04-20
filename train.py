import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import os

from model import PixelCNN
from dataset import get_data_loader

# --- Configuration ---
DATA_DIR = './data'
MODEL_SAVE_PATH = './saved_models/pixelcnn_bedroom.pth'
IMAGE_SIZE = 64
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50 # Adjust as needed, more epochs yield better results
N_FILTERS = 128
N_BLOCKS = 5
OUTPUT_BINS = 256 # For 8-bit color depth

def train():
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs('./saved_models', exist_ok=True)
    
    # --- Data ---
    dataloader = get_data_loader(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
    
    # --- Model ---
    model = PixelCNN(in_channels=3, n_filters=N_FILTERS, n_blocks=N_BLOCKS, output_bins=OUTPUT_BINS).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for occluded_imgs, original_imgs in progress_bar:
            occluded_imgs = occluded_imgs.to(device)
            original_imgs = original_imgs.to(device)
            
            # The target for the loss function should be integer class labels
            # We convert the [0,1] float image to [0, 255] integer image
            target = (original_imgs * (OUTPUT_BINS - 1)).long()

            # --- Forward Pass ---
            optimizer.zero_grad()
            output = model(occluded_imgs) # Output shape: (B, bins, C, H, W)
            
            # --- Calculate Loss ---
            # We need to compute loss for each channel (R,G,B) separately
            # and then average them.
            loss_r = criterion(output[:, :, 0, :, :], target[:, 0, :, :])
            loss_g = criterion(output[:, :, 1, :, :], target[:, 1, :, :])
            loss_b = criterion(output[:, :, 2, :, :], target[:, 2, :, :])
            loss = (loss_r + loss_g + loss_b) / 3.0
            
            # --- Backward Pass ---
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

    # --- Save Model ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()