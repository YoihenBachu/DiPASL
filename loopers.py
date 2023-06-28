import torch
import os
from tqdm import tqdm
import warnings
import wandb
from sklearn.metrics import accuracy_score

warnings.filterwarnings(action = "ignore")

def looper(epochs,
           optimizer,
           model,
           trainloader,
           testloader,
           loss_fn,
           device,
           wandb_log,
           model_savepath,
           bname,
           optim,
           lr
):
    e = 0
    for epoch in range(epochs):
        e += 1
        print('Epoch {}/{}, lr:{}'.format(epoch + 1, epochs, optimizer.param_groups[0]['lr']))
        print('-' * 30)
        # Train the model
        train_loss = 0.0
        train_predictions = []
        train_labels = []
        model.train()  # Set the model to training mode
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for (_, data) in pbar:
            pbar.set_description(f"Epoch {epoch+1} | Learning Rate: {optimizer.param_groups[0]['lr']}")
            images = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()

            out = model(images)
            truth = torch.max(labels, dim=1)[1]
            loss = loss_fn(out, truth)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(truth.cpu().numpy())

            pbar.set_postfix({'TrainLoss': loss.item()})
            if wandb_log:
                wandb.log({"Training loss": loss.item()})

        train_acc = accuracy_score(train_labels, train_predictions)
        train_loss = train_loss / len(trainloader)

        # Evaluate the model
        test_loss = 0.0
        test_predictions = []
        test_labels = []
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            pbar = tqdm(enumerate(testloader), total=len(testloader))
            for (_, data) in pbar:
                pbar.set_description(f"Epoch {epoch+1} | Learning Rate: {optimizer.param_groups[0]['lr']}")
                images = data[0].to(device)
                labels = data[1].to(device)

                out = model(images)
                truth = torch.max(labels, 1)[1]
                loss = loss_fn(out, truth)

                test_loss += loss.item()
                _, predicted = torch.max(out.data, 1)
                test_predictions.extend(predicted.cpu().numpy())
                test_labels.extend(truth.cpu().numpy())

                pbar.set_postfix({'TestLoss': loss.item()})
                if wandb_log:
                    wandb.log({"Testing loss": loss.item()})
                    wandb.log({"Training accuracy": train_acc})

        test_acc = accuracy_score(test_labels, test_predictions)
        test_loss = test_loss / len(testloader)

        filename = str(bname) + '_' + str(optim) + '_' + str(lr) + '_' + str(epoch+1) + '.pt'
        if wandb_log:
            if ((e) >= 15) and ((e)%5 == 0):
                torch.save(model.state_dict(), os.path.join(model_savepath, str(filename)))
                wandb.alert(
                    title = 'Update',
                    text = f'Epoch: {epoch+1}\nTraining Loss: {train_loss} \nValidation Loss: {test_loss} \nAccuracy: {train_acc} \nModel saved at {filename}',
                )
            else:
                wandb.alert(
                    title = 'Update',
                    text = f'Epoch: {epoch+1}\nTraining Loss: {train_loss} \nValidation Loss: {test_loss} \nAccuracy: {train_acc}',
            )

        else:
            if ((e) >= 15) and ((e)%5 == 0):
                torch.save(model.state_dict(), os.path.join(model_savepath, str(filename)))
                
        # Print the training and testing statistics
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.2f}%, Test Loss: {:.4f}, Test Acc: {:.2f}%'
            .format(epoch+1, epochs, train_loss, train_acc, test_loss, test_acc))
