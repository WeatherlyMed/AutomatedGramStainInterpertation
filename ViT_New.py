import timm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import ImageFolder

print(timm.list_models("crossvit*"))
num_classes = 3

model = timm.create_model("crossvit_15_dagger_240", pretrained=False)

#model = CrossViT(image_size=240, num_classes=num_classes, pretrained=True).to(device)
model.head[0] = nn.Linear(in_features=192, out_features=num_classes, bias=True)
model.head[1] = nn.Linear(in_features=384, out_features=num_classes, bias=True)

#weights_path = 'C:\\Users\\878074\\RSEF\\crossvit_15_dagger_224.pth'
#checkpoint = torch.load(weights_path)
#model.load_state_dict(checkpoint)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(model)

transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

pin_memory = True if device.type == 'cuda' else False

# Update your DataLoader to utilize pin_memory
train_dataset = ImageFolder(root='C:\\Users\\878074\\RSEF\\NewSorted', transform=transform)
data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=pin_memory)
# Validation dataset
val_dataset = ImageFolder(root='C:\\Users\\878074\\RSEF\\val\\val', transform=transform)
val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=pin_memory)


class_weights = torch.tensor([1.0, 1.5, 2.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

num_epochs = 120  # Adjust based on performance

# Create DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
   
    loop = tqdm(train_loader, leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=running_loss/total, acc=correct/total)

    # Evaluate on validation set
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {correct/total:.4f}")
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Acc: {val_correct/val_total:.4f}")

torch.save(model.state_dict(), "cross_gramstain.pth")