import torch
import wandb
from ray import train

# Define the training function
def train_autoencoder(model, train_dataloader, val_dataloader, criterion, optimizer, metric, device, num_epochs = 10):

    train_loss_vec = []
    val_loss_vec = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_train_metric = 0.0

        for data in train_dataloader:
            images = data
            images = images.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, images)

            # Compute the metric
            train_metric = metric(outputs, images)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_train_metric += train_metric.item()

            total_train_loss += loss.item()
            train_loss_vec.append(loss.item())


        model.eval()
        total_val_loss = 0.0
        total_val_metric = 0.0

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

                # Compute the validation metric
                val_metric = metric(val_outputs, val_images)

                total_val_metric += val_metric.item()
                total_val_loss += val_loss.item()
                val_loss_vec.append(loss.item())

        
        # Print average loss and MSE for the epoch
        average_train_loss = total_train_loss / len(train_dataloader)
        average_train_metric = total_train_metric / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_train_loss}, Train Metric: {average_train_metric}")

        average_val_loss = total_val_loss / len(val_dataloader)
        average_val_metric = total_val_metric / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {average_val_loss}, Val Metric: {average_val_metric}")

        # Log metrics to Weights and Biases
        #wandb.log({"epoch": epoch, "train_loss": average_train_loss, "val_loss": average_val_loss, "train_metric": average_train_metric, "val_metric": average_val_metric})

        #if epoch % 5 == 0:
        #wandb.log({"epoch": epoch, "train_loss": average_train_loss, "val_loss": average_val_loss, "train_metric": average_train_metric, "val_metric": average_val_metric})
        train.report({"epoch": epoch, "train_loss": average_train_loss, "val_loss": average_val_loss, "train_metric": average_train_metric, "val_metric": average_val_metric})
            
    return average_train_loss, average_val_loss, train_loss_vec, val_loss_vec