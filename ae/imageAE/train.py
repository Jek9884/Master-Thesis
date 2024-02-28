import torch
import wandb
from ray import train

# Define the training function
def train_autoencoder(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs = 10):

    # Init Weights and Biases
    #wandb.init(project="Master-Thesis", name="Pong-AE")

    train_loss_vec = []
    val_loss_vec = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for data in train_dataloader:
            images = data
            images = images.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, images)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_loss_vec.append(loss.item())


        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for val_data in val_dataloader:
                val_images = val_data
                for image in val_images:
                    image = image.unsqueeze(0)
                val_images = val_images.to(device)

                #Forward pass
                val_outputs = model(val_images)

                # Compute the validation loss
                val_loss = criterion(val_outputs, val_images)
                total_val_loss += val_loss.item()
                val_loss_vec.append(loss.item())

        
        # Print average loss for the epoch
        average_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss:.4f}")
        
        average_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {average_val_loss:.4f}")

        #wandb.log({"epoch": epoch, "train_loss": average_train_loss, "val_loss": average_val_loss})
        train.report({"epoch": epoch, "train_loss": average_train_loss, "val_loss": average_val_loss})
    return average_train_loss, average_val_loss, train_loss_vec, val_loss_vec