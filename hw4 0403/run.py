import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import copy
import sys
from utils import load_image, Normalization, device, imshow, get_image_optimizer, unify_shape
from style_and_content import ContentLoss, StyleLoss


"""A ``Sequential`` module contains an ordered list of child modules. For
instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU, MaxPool2d,
Conv2d, ReLU…) aligned in the right order of depth. We need to add our
content loss and style loss layers immediately after the convolution
layer they are detecting. To do this we must create a new ``Sequential``
module that has content loss and style loss modules correctly inserted.
"""

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_model_and_losses(cnn, style_img, content_img,
                               content_layers:list=content_layers_default,
                               style_layers:list=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # build a sequential model consisting of a Normalization layer
    # then all the layers of the VGG feature network along with ContentLoss and StyleLoss
    # layers in the specified places

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # here if you need a nn.ReLU layer, make sure to use inplace=False
    # as the in place version interferes with the loss layers
    # trim off the layers after the last content and style losses
    # as they are vestigial

    normalization = Normalization().to(device)
    style_img = normalization(style_img)
    content_img = normalization(content_img)
    
    layers = []
    conv_layer_count = 0
    layer: nn.Module
    for layer in cnn.children():
        if isinstance(layer, nn.ReLU):
            layer.inplace = False

        layers.append(layer)
        style_img = layer(style_img)
        content_img = layer(content_img)

        if isinstance(layer, nn.Conv2d):
            conv_layer_count += 1
            layer_name = f"conv_{conv_layer_count}"

            if layer_name in content_layers:
                content_layers.remove(layer_name)
                loss_layer = ContentLoss(content_img)
                content_losses.append(loss_layer)
                layers.append(loss_layer)
                
            if layer_name in style_layers:
                style_layers.remove(layer_name)
                loss_layer = StyleLoss(style_img)
                style_losses.append(loss_layer)
                layers.append(loss_layer)

            if not (content_layers or style_layers):
                break


    model = nn.Sequential(*layers)
    print(model)

    # raise NotImplementedError()

    return model, style_losses, content_losses


"""Finally, we must define a function that performs the neural transfer. For
each iteration of the networks, it is fed an updated input and computes
new losses. We will run the ``backward`` methods of each loss module to
dynamicaly compute their gradients. The optimizer requires a “closure”
function, which reevaluates the module and returns the loss.

We still have one final constraint to address. The network may try to
optimize the input with values that exceed the 0 to 1 tensor range for
the image. We can address this by correcting the input values to be
between 0 to 1 each time the network is run.



"""


def run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True, num_steps=300,
                     style_weight=1000000, content_weight=1):
    """Run the image reconstruction, texture synthesis, or style transfer."""
    print('Building the style transfer model..')
    # get your model, style, and content losses
    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img, style_layers=[]) # type: tuple[nn.Sequential, list, list]
    # get the optimizer
    model.requires_grad_(False)
    opt = get_image_optimizer(input_img)
    
    # run model training, with one weird caveat
    # we recommend you use LBFGS, an algorithm which preconditions the gradient
    # with an approximate Hessian taken from only gradient evaluations of the function
    # this means that the optimizer might call your function multiple times per step, so as
    # to numerically approximate the derivative of the gradients (the Hessian)
    # so you need to define a function
    # here
    # which does the following:
    # clear the gradients
    # compute the loss and it's gradient
    # return the loss

    # one more hint: the images must be in the range [0, 1]
    # but the optimizer doesn't know that
    # so you will need to clamp the img values to be in that range after every step
    
    for _ in range(num_steps):
        
        def closure():
            '''a closure that reevaluates the model and returns the loss'''
            nonlocal input_img

            opt.zero_grad()
            model(input_img)

            total_loss: torch.Tensor = 0
            if use_content:
                layer: ContentLoss
                for layer in content_losses:
                    total_loss += layer.loss * content_weight
            if use_style:
                layer: StyleLoss
                for layer in style_losses:
                    total_loss += layer.loss * style_weight
            total_loss.backward()

            with torch.no_grad():
                input_img = torch.clamp_(input_img, 0, 1)

            return total_loss
        
        orig_loss = opt.step(closure)
        print(orig_loss)
        
    input_img = torch.clamp_(input_img, 0, 1)
    return input_img


def main(style_img_path, content_img_path):
    # we've loaded the images for you
    style_img = load_image(style_img_path)
    content_img = load_image(content_img_path)

    # interative MPL
    plt.ion()

    style_img_original = style_img
    style_img = unify_shape(content_img, style_img)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # plot the original input image:
    plt.figure()
    imshow(style_img_original, title='Style Image Original')
    plt.figure()
    imshow(style_img, title='Style Image Resized')

    plt.figure()
    imshow(content_img, title='Content Image')


    # we load a pretrained VGG19 model from the PyTorch models library
    # but only the feature extraction part (conv layers)
    # and configure it for evaluation
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # image reconstruction
    print("Performing Image Reconstruction from white noise initialization")
    # input_img = random noise of the size of content_img on the correct device
    input_img = torch.rand_like(content_img, requires_grad=True, device=device)
    # output = reconstruct the image from the noise
    output = run_optimization(cnn, content_img, style_img, input_img)

    plt.figure()
    imshow(output, title='Reconstructed Image')
    plt.ioff()
    plt.show()
    return 

    # texture synthesis
    print("Performing Texture Synthesis from white noise initialization")
    # input_img = random noise of the size of content_img on the correct device
    # output = synthesize a texture like style_image

    plt.figure()
    imshow(output, title='Synthesized Texture')

    # style transfer
    # input_img = random noise of the size of content_img on the correct device
    # output = transfer the style from the style_img to the content image

    plt.figure()
    imshow(output, title='Output Image from noise')

    print("Performing Style Transfer from content image initialization")
    # input_img = content_img.clone()
    # output = transfer the style from the style_img to the content image

    plt.figure()
    imshow(output, title='Output Image from noise')

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    args = sys.argv[1:3]
    main(*args)
