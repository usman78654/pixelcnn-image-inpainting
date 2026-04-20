import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageCompletionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.occluded_dir = os.path.join(root_dir, 'train', 'occluded_images')
        self.original_dir = os.path.join(root_dir, 'train', 'original_images')
        
        self.occluded_images = sorted([f for f in os.listdir(self.occluded_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.original_images = sorted([f for f in os.listdir(self.original_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        # --- ADD THIS NEW BLOCK ---
        if len(self.occluded_images) != len(self.original_images):
            raise ValueError(
                f"Mismatch in number of images found!\n"
                f"Found {len(self.occluded_images)} images in {self.occluded_dir}\n"
                f"Found {len(self.original_images)} images in {self.original_dir}\n"
                f"Please ensure both directories have the same number of images."
            )
        # --- END OF NEW BLOCK ---
        self.transform = transform

    def __len__(self):
        return len(self.occluded_images)

    def __getitem__(self, idx):
        occluded_img_path = os.path.join(self.occluded_dir, self.occluded_images[idx])
        original_img_path = os.path.join(self.original_dir, self.original_images[idx])

        occluded_image = Image.open(occluded_img_path).convert('RGB')
        original_image = Image.open(original_img_path).convert('RGB')

        if self.transform:
            occluded_image = self.transform(occluded_image)
            original_image = self.transform(original_image)
            
        return occluded_image, original_image

def get_data_loader(data_dir, batch_size=32, image_size=64):
    """
    Creates and returns a DataLoader for the image completion dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), # Scales images to [0.0, 1.0]
    ])
    
    dataset = ImageCompletionDataset(root_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader