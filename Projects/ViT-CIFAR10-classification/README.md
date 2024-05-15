# ViT

## Introduction

This is a simple implementation of the Vision Transformer (ViT) model for digit recognition. The model is trained on the MNIST dataset. The model is implemented using PyTorch.

## Model

The model is a simple implementation of the Vision Transformer (ViT) model. The model consists of a patch embedding layer, a transformer encoder, and a linear layer. The patch embedding layer converts the input image into a sequence of patches. The transformer encoder processes the sequence of patches. The linear layer is used to classify the input image.

## Training

The model is trained on the MNIST dataset. The model is trained using the Adam optimizer with a learning rate of 0.0001. The model is trained for 10 epochs.

## Results

The model achieves an accuracy of 98.5% on the MNIST test dataset.

## Usage

To train the model, run the following command:

```
python train.py
```

To evaluate the model, run the following command:

```
python evaluate.py
```

# Results

| Model        | Accuracy (400 epochs) |
| ------------ | --------------------- |
| ViT(patch=4) | 84.46%                |
| ResNet18     | 93.87%                |
| ViT_tiny     | 94.00%                |

## References

1. [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
2. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openai.com/blog/dall-e/)
3. [PyTorch](https://pytorch.org/)
4. [MNIST](http://yann.lecun.com/exdb/mnist/)
