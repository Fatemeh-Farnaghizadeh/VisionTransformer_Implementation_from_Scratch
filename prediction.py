import torch
import models
import utils
import dataset

if __name__ == '__main__':

    test_data, _ = dataset.get_test_loader()

    trained_model = models.ViT().to(utils.DEVICE)

    model = utils.load_model(model=trained_model, model_name='ViT_Model.pth')

    model.eval()

    for idx, (images, labels) in enumerate(test_data):
        if idx == 0:
            images = images.to(utils.DEVICE)
            labels = labels.to(utils.DEVICE)

            with torch.no_grad():
                pred = model(images)
        
            y_hat = torch.argmax(pred, dim=1)
        
            for i in range(len(y_hat)):
                print(f"Predicted: {y_hat[i].item()}, Ground Truth: {labels[i].item()}")




