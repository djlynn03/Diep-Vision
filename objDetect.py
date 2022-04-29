from detecto import core, utils
from detecto.visualize import show_labeled_image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(random.randint(1,359)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    utils.normalize_transform()
])

Train_dataset = core.Dataset('trainingFiles/', transform=custom_transforms)
Test_dataset = core.Dataset('testFiles/')
loader = core.DataLoader(Train_dataset, batch_size=1, shuffle=True)

def train(lr=.0001, epochs=10, show=False, forceSave=False):
    try:
        minLoss = float(open("minLoss.txt", "r").readline())
    except:
        minLoss = 10

    try:
        model = core.Model.load("model_weights.pth", ["Square", "Triangle", "Pentagon", "Player"])
    except:
        model = core.Model(["Square", "Triangle", "Pentagon", "Player"])
    
    def randomizeParameters():
        return [random.uniform(0.01,5.00),
                lr,
                random.random(),
                random.random(),
                random.random()]
    
    # parameters = randomizeParameters()
    # lrStep, lr, m, wd, g = parameters
    lrStep, lr, m, wd, g = parameters = [1, lr, 0.9, 0.0005, 0.1]
    
    losses = model.fit(loader, Test_dataset, epochs=epochs, lr_step_size=lrStep, learning_rate=lr, momentum=m, weight_decay=wd, gamma=g, verbose=True)
    torch.cuda.empty_cache()

    def showResults():
        plt.title("Loss")
        plt.plot(losses)
        plt.show()

        image = utils.read_image("testFiles/page5.png") 
        predictions = model.predict(image)
        labels, boxes, scores = predictions
        thresh = 0.0
        filtered_indices = np.where(scores>thresh)
        filtered_scores = scores[filtered_indices]
        filtered_boxes = boxes[filtered_indices]
        num_list = filtered_indices[0].tolist()
        filtered_labels = [labels[i] for i in num_list]
        combined_labels = [tup[0] + " " + tup[1] for tup in zip(filtered_labels, [str(round(filtered_scores.data[i].item(), 2)) for i in range(len(filtered_scores))])]
        show_labeled_image(image, filtered_boxes, combined_labels)

    if forceSave:
        model.save("model_weights.pth")
        print("Saved Model")
    elif losses[-1] <= minLoss:
        model.save("model_weights.pth")
        open("minLoss.txt", "w").write(str(losses[-1]))
        open("parameters/bestParameters.txt", "a").write(str(losses[-1]) + " " + str(parameters) + "\n")
        print("Saved New Best Model")
    
    # open("parameters/losses.txt", "a").write(str(losses[-1]) + "\n")
    # open("parameters/lrSteps.txt", "a").write(str(lrStep) + "\n")
    # open("parameters/lrs.txt", "a").write(str(lr) + "\n")
    # open("parameters/moments.txt", "a").write(str(m) + "\n")
    # open("parameters/wds.txt", "a").write(str(wd) + "\n")
    # open("parameters/gs.txt", "a").write(str(g) + "\n")
    
    if show == True:
        showResults()
        
    return (losses[-1], minLoss)

def recTrain(lr, cycles=0, epochs=1):
    print("Cycle:", cycles + 1)
    print("Learning Rate:", lr)
    trainData = train(lr, epochs)
    if trainData[0] <= trainData[1]:
        recTrain(lr)
    else:
        if cycles >= 9:
            if random.random() < .05:
                recTrain(lr * 5)
            else:
                recTrain(lr / 5)
        else:
            recTrain(lr, cycles=cycles+1)

def infTrain(lr, epochs=1):
    while True:
        train(lr, epochs)

def randTrain():
    while True:
        lr = random.uniform(.0001,.01)
        train(lr, 1)

train(.001, 1000, True, False)
# recTrain(.01)
# infTrain(.001, 10)
# randTrain()