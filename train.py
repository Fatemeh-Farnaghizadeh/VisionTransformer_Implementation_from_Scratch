import utils
import dataset
import models
import torch

from torch import optim, nn

if __name__ == "__main__":

    train_loader, _ = dataset.get_train_loader()
    test_loader, _ = dataset.get_test_loader()

    model = models.ViT().to(utils.DEVICE)

    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=utils.LR,
        weight_decay=utils.WEIGHT_DECAY
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(utils.EPOCHS):
        
        if epoch % 2 == 0:
            utils.save_model(model=model, model_name='ViT_Model.pth')

        train_loss = 0
        test_loss = 0

        train_acc = 0
        test_acc = 0

        for imgs, labels in train_loader:

            imgs = imgs.to(utils.DEVICE)
            labels = labels.to(utils.DEVICE)

            y_hat = model(imgs)

            optimizer.zero_grad()

            loss = loss_fn(y_hat, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss

            acc = (
                (y_hat.argmax(dim=1) == labels).sum().item()/len(labels)
            )

            train_acc += acc

        train_acc = train_acc / len(train_loader)
        train_loss = train_loss / len(train_loader)

        print(f'EPOCH: {epoch}: train_loss={train_loss}  train_acc={train_acc}')

        model.eval()

        for img_test, labels_test in test_loader:

            img_test = img_test.to(utils.DEVICE)
            labels_test = labels_test.to(utils.DEVICE)

            with torch.no_grad():
                y_hat_test = model(img_test)
        
            loss = loss_fn(y_hat_test, labels_test)
            test_loss += loss

            acc = (
                (y_hat_test.argmax(dim=1) == labels_test).sum().item()/len(labels_test)
            )
            test_acc += acc
        
        test_acc = test_acc / len(test_loader)
        test_loss = test_loss / len(test_loader)

        print(f'EPOCH: {epoch}: test_loss={test_loss}  test_acc={test_acc}')