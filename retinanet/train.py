import torch
    
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import transform as T
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import SGD
import torchvision
from dataset import TextDataset
import time 

#https://pseudo-lab.github.io/Tutorial-Book-en/chapters/en/object-detection/Ch4-RetinaNet.html

# Définir la fonction de chargement de données
def collate_fn(batch):
    return tuple(zip(*batch))


torch.cuda.empty_cache()

dataset = TextDataset("../PAD/trainval.txt", image_dir="../PAD")
dataset.filter()
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn)


retina = torchvision.models.detection.retinanet_resnet50_fpn(num_classes = 1, pretrained=False, pretrained_backbone = True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_epochs = 5
retina.to(device)
    
# parameters
params = [p for p in retina.parameters() if p.requires_grad] # select parameters that require gradient calculation
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)



# about 4 min per epoch on Colab GPU
for epoch in range(num_epochs):
    start = time.time()
    retina.train()

    i = 0    
    epoch_loss = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = retina(images, targets) 
        losses = sum(loss for loss in loss_dict.values()) 
        i += 1
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print(torch.cuda.memory_allocated())
        epoch_loss += losses
    print(epoch_loss, f'time: {time.time() - start}')

torch.save(retina.state_dict(),f'retina_{num_epochs}.pt')