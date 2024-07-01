from torchstat import stat
from torchvision.models import resnet18

model = resnet18()
stat(model, (3, 224, 224))