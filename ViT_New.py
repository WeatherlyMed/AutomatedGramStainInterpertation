import timm  # Library for pre-trained deep learning models
import torch
import torch.nn as nn  # Neural network module
from torchvision import transforms  # Data preprocessing and augmentation
from torch.optim import AdamW  # Optimizer
from torch.utils.data import DataLoader  # Data loading utilities
from tqdm import tqdm  # Progress bar for loops
from torchvision.datasets import ImageFolder  # For loading datasets organized in folders

# Print available models that match the "crossvit*" pattern
print(timm.list_models("crossvit*"))

# Define the number of output classes
num_classes = 3

# Create a CrossViT model (variant: crossvit_15_dagger_240) without pre-trained weights
model = timm.create_model("crossvit_15_dagger_240", pretrained=False)

# Modify the final classification heads to match the number of output classes
model.head[0] = nn.Linear(in_features=192, out_features=num_classes, bias=True)
model.head[1] = nn.Linear(in_features=384, out_features=num_classes, bias=True)

# Uncomment the lines below to load pre-trained weights if available
# weights_path = 'C:\\Users\\878074\\RSEF\\crossvit_15_dagger_224.pth'
# checkpoint = torch.load(weights_path)
# model.load_state_dict(checkpoint)

# Define the device: use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Move model to the selected device
print(model)

# Define transformations for preprocessing the input images
transform = transforms.Compose([
    transforms.Resize((240, 240)),  # Resize images to 240x240
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values
])

# Use pinned memory for faster data transfer to GPU if CUDA is available
pin_memory = True if device.type == 'cuda' else False

# Load the training dataset
train_dataset = ImageFolder(root='C:\\Users\\878074\\RSEF\\NewSorted', transform=transform)

# Create a DataLoader for the training dataset
data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=pin_memory)

# Load the validation dataset
val_dataset = ImageFolder(root='C:\\Users\\878074\\RSEF\\val\\val', transform=transform)

# Create a DataLoader for the validation dataset
val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=pin_memory)

# Define class weights to address class imbalance during training
class_weights = torch.tensor([1.0, 1.5, 2.0]).to(device)

# Use CrossEntropyLoss with class weights for multi-class classification
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Define the AdamW optimizer with learning rate and weight decay
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

# Set the number of epochs for training
num_epochs = 120  # Adjust this value based on performance and dataset size

# Create DataLoader for training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Start the training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss, correct, total = 0.0, 0, 0  # Initialize metrics for the epoch
    
    # Create a progress bar for the training loop
    loop = tqdm(train_loader, leave=True)
    for images, labels in loop:
        # Move images and labels to the device (GPU/CPU)
        images, labels = images.to(device), labels.to(device)
        
        # Zero the gradients from the previous step
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        outputs = model(images)
        
        # Compute the loss
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Update running metrics
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)  # Get predicted class
        correct += (preds == labels).sum().item()  # Count correct predictions
        total += labels.size(0)  # Total number of samples
        
        # Update the progress bar with loss and accuracy
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=running_loss/total, acc=correct/total)

    # Evaluate the model on the validation set
    model.eval()  # Set the model to evaluation mode
    val_loss, val_correct, val_total = 0.0, 0, 0  # Initialize validation metrics

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for images, labels in val_loader:
            # Move images and labels to the device (GPU/CPU)
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass: compute predictions
            outputs = model(images)
            
            # Compute the loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Update validation metrics
            preds = outputs.argmax(dim=1)  # Get predicted class
            val_correct += (preds == labels).sum().item()  # Count correct predictions
            val_total += labels.size(0)  # Total number of samples

    # Print training and validation metrics for the epoch
    print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {correct/total:.4f}")
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Acc: {val_correct/val_total:.4f}")

# Save the trained model's state dictionary to a file
torch.save(model.state_dict(), "cross_gramstain.pth")
