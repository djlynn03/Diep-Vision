import torch
import torchvision.models as models
import polygon_nn
from torch import nn
import polygon_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

# model = polygon_nn.NeuralNetwork()

# torch.save(model.state_dict(), 'model_weights.pth')

# model = polygon_nn.NeuralNetwork()

# torch.save(model.state_dict(), 'model_weights.pt')
# model = NeuralNetwork().to(device)

device = "cuda" if torch.cuda.is_available() else "cpu"

learning_rate = 1e-2
batch_size = 3
# epochs = 5

train_dataloader = DataLoader(polygon_dataset.PolygonImageDataset(annotations_file='.\\images\\labels.csv', img_dir='.\\images'), batch_size=64, shuffle=True)
print("Train: ", len(train_dataloader.dataset))
test_dataloader = DataLoader(polygon_dataset.PolygonImageDataset(annotations_file='.\\test_images\\labels.csv', img_dir='.\\test_images'), batch_size=64, shuffle=True)
print("Test: ", len(test_dataloader.dataset))

model = polygon_nn.NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 100
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     polygon_nn.train_loop(train_dataloader, model, loss_fn, optimizer)
#     polygon_nn.test_loop(test_dataloader, model, loss_fn)
# print("Done!")

# torch.save(model.state_dict(), 'model_weights.pt')

model.load_state_dict(torch.load('model_weights.pt'))
model.eval()


img = Image.open('.\\all_images\\pentagon\\pentagon3.png')
img = img.resize((64,64))
img = img.convert('RGB')
img = transforms.ToTensor()(img)
img = img.unsqueeze(0)
output = model(img.to(device))

prediction = int(torch.max(output.data, 1)[1].numpy())

dataset = datasets.ImageFolder('.\\all_images\\', transform=transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()]))

print(dataset.class_to_idx, dataset.classes)
print(prediction)
if prediction == 2:
    print("Prediction: Pentagon")
elif prediction == 1:
    print("Prediction: Triangle")
elif prediction == 0:
    print("Prediction: Square")
    
# epochs = 100
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     polygon_nn.train_loop(train_dataloader, model, loss_fn, optimizer)
#     polygon_nn.test_loop(test_dataloader, model, loss_fn)
# print("Done!")

# loaded_model = polygon_nn.NeuralNetwork()

# loaded_model.load_state_dict(torch.load('model_weights.pt'))
# print(loaded_model.)