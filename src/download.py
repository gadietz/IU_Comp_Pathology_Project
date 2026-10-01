import os
import medmnist
from medmnist import INFO

# Set the data flag
data_flag = 'pneumoniamnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

# Save files
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

print(f"Downloading {info['python_class']} dataset...")
    
# Download train, validation, and test splits
train_dataset = DataClass(split='train', download=True, root=output_dir)
val_dataset = DataClass(split='val', download=True, root=output_dir)
test_dataset = DataClass(split='test', download=True, root=output_dir)
    
print(f"Successfully downloaded PneumoniaMNIST to '{output_dir}/'")