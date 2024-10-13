# NSFW-MobileNet
### Introduction
Train your own NSFW model with the 'torchvision.models.mobilenet_v3_large' pre-trained model.

I found that most other NSFW projects simply resize the images to 224x224, but the original images are usually not square, it might be a wrong way to resize non-square images to square images, the resized images should keep their own weight/height ratio not changed.

This project always resizes images with their own weight/height ratio not changed, and use multiple overlapped image patches to inference, the result should be better.
### How To Train
1. Create the './checkpoint' subdirectory and the './data/train' subdirectory manually.
2. Copy your own NSFW dataset to the './data/train' subdirectory, the NSFW dataset should be in several subdirectories, such as 'drawings', 'hentai', 'neutral', 'porn', 'sexy', but they can be any other subdirectories, such as 'nude', 'safe', 'sexy'.
3. Run 'python3 train.py'.
### Dataset
I had trained the model with this dataset: [https://huggingface.co/datasets/deepghs/nsfw_detect](https://huggingface.co/datasets/deepghs/nsfw_detect)

Note: The link was gone on the official website, please download the dataset from other huggingface mirror websites.

If you want to get more accurate result, you can use [https://github.com/infinitered/nsfwjs](https://github.com/infinitered/nsfwjs) to classify all the images in the above dataset, and train the model with the refined dataset from scratch.
### Pre-Trained Model
The pre-trained models are 'model_full.pth' and 'model_full2.pth, the first one had been trained by the original dataset, while the second one had been trained by the refined dataset, you can use either of the pre-trained models for any purposes, including commercial projects.
### How To Inference
Since the input of the mobilenet_v3_large model is 224x224, but the image is usually not square, I suggest that you resize the image first, let the smaller edge of the image match 224, then crop the resized image to three overlapped patches of 224x224, inference the patches separately, if any of the result is NSFW, the final result should be NSFW.
### License
The MIT License (MIT)
