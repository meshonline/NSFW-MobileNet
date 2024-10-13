import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

model = torch.load('./model_full.pth')
model = model.to(torch.device("cpu"))
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    normalize,])

img_rgb = Image.open('./image.jpg', mode='r').convert('RGB')
img_rgb = transform(img_rgb).unsqueeze(0)
# classes=['drawings', 'hentai', 'neutral', 'porn', 'sexy']
img_rgb_list = []
if img_rgb.shape[2] == 224:
    left = 0
    right = left + 224
    top = 0
    bottom = top + 224
    img_rgb_list.append(img_rgb[:, :, top:bottom, left:right])
    left = (img_rgb.shape[3] - 224) // 2
    right = left + 224
    img_rgb_list.append(img_rgb[:, :, top:bottom, left:right])
    left = img_rgb.shape[3] - 224
    right = left + 224
    img_rgb_list.append(img_rgb[:, :, top:bottom, left:right])
else:
    left = 0
    right = left + 224
    top = 0
    bottom = top + 224
    img_rgb_list.append(img_rgb[:, :, top:bottom, left:right])
    top = (img_rgb.shape[2] - 224) // 2
    bottom = top + 224
    img_rgb_list.append(img_rgb[:, :, top:bottom, left:right])
    top = img_rgb.shape[2] - 224
    bottom = top + 224
    img_rgb_list.append(img_rgb[:, :, top:bottom, left:right])
is_nsfw = False
for img_rgb in img_rgb_list:
    pred = model(img_rgb)
    pred[0] = F.softmax(pred[0], dim=0)
    class_index = torch.argmax(pred).item()
    if (class_index == 1 or class_index == 3) and pred[0][class_index].item() > 0.8:
        is_nsfw = True
        break
print(is_nsfw)
