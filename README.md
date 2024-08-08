# NSFW-MobileNet
### Introduction
You can train your own NSFW model with the 'torchvision.models.mobilenet_v3_small' pretrained model in DDP mode, and you can use the pre-trained model for free, including any commercial purposes.
### How To Train
1. Create the './checkpoint' subdirectory and the './data/train' subdirectory manually.
2. Copy your own NSFW dataset to the './data/train' subdirectory, the NSFW dataset should be in several subdirectories, such as 'drawings', 'hentai', 'neutral', 'porn', 'sexy', but they can be any other subdirectories.
3. Run 'python3 train.py'.
### Dataset
I had trained the model with the dataset: [https://huggingface.co/datasets/deepghs/nsfw_detect](https://huggingface.co/datasets/deepghs/nsfw_detect)
### How To Inference
Since the input size of the mobilenet_v3_small model is 224x224, but the image to inference is usually not square, I suggest that you resize the image first, let the smaller edge of the image match 224, then crop the resized image to two or three 224x224 patches, inference the patches separately, if any of the inference result is NSFW, the final inference result should be NSFW, for example:
```python
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
# classes=['drawing', 'hentai', 'neutral', 'porn', 'sexy']
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
```
### License
The MIT License (MIT)
