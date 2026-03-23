# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset

This experiment demonstrates transfer learning using a pre-trained ResNet18 model on a custom image dataset. Instead of training a deep neural network from scratch, the pre-trained model’s feature extraction layers are reused, and only the final classification layer is retrained. This approach reduces training time, requires less data, and achieves high accuracy.

## DESIGN STEPS
### STEP 1:
Data Preprocessing – Resize all images to 224×224 and convert them into tensors suitable for ResNet input.

### STEP 2: 
Dataset Loading – Organize images into train/test sets and load them using ImageFolder and DataLoader.

### STEP 3:
Load Pretrained Model – Use ResNet18 trained on ImageNet as the base model.

### STEP 4:
Modify Final Layer – Freeze earlier layers and replace the fully connected layer to match the number of dataset classes.

### STEP 5:
Train and Evaluate – Train only the final layer, then test the model and analyze results using a confusion matrix and classification report.

## PROGRAM
```python
# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

# Modify the final fully connected layer to match the dataset classes
for param in model.parameters():
    param.requires_grad = False   # freeze earlier layers
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

```
```python
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:bala murugan s")
    print("Register Number:  212223230027")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="953" height="711" alt="Screenshot 2026-03-23 211143" src="https://github.com/user-attachments/assets/cda1ccc7-10e0-48a2-b85f-46e593e88a9b" />


### Confusion Matrix
<img width="574" height="476" alt="Screenshot 2026-03-23 211342" src="https://github.com/user-attachments/assets/1c05b16f-dd46-4499-8510-71876902395c" />


### Classification Report
<img width="436" height="195" alt="Screenshot 2026-03-23 211405" src="https://github.com/user-attachments/assets/79c82f74-f583-4bd8-9b97-20ef445d2fe6" />


### New Sample Prediction
<img width="442" height="362" alt="Screenshot 2026-03-23 211502" src="https://github.com/user-attachments/assets/941e1ef2-aa5d-49d3-925e-2f75829f5d57" />

<img width="434" height="360" alt="Screenshot 2026-03-23 211515" src="https://github.com/user-attachments/assets/e0fb077a-5c97-4d86-8383-b07619ee2812" />

## RESULT
The Implementation of Transfer Learning for classification using VGG-19 architecture is successful.
