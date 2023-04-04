import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import copy
import sys
from utils import load_image, Normalization, device, imshow, get_image_optimizer, unify_shape
from style_and_content import ContentLoss, StyleLoss
import argparse


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
                               style_layers:list=style_layers_default, **kwargs):
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
    
    layers = [normalization]
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
                     style_weight=1000000, content_weight=1, **kwargs):
    """Run the image reconstruction, texture synthesis, or style transfer."""
    print('Building the style transfer model..')
    # get your model, style, and content losses
    # if use_content and kwargs.get("content_layer") is not None:
    #     kwargs["content_layers"] = [f'conv_{kwargs["content_layer"]}']
    if not use_content:
        kwargs["content_layers"] = []
    if not use_style:
        kwargs["style_layers"] = []

    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img, **kwargs) # type: tuple[nn.Sequential, list, list]
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

            # with torch.no_grad():
            #     input_img = torch.clamp_(input_img, 0, 1)

            return total_loss
        
        orig_loss = opt.step(closure)
        print(orig_loss)
    
    with torch.no_grad():
        input_img = torch.clamp_(input_img, 0, 1)
    return input_img


def main(style_img_path, content_img_path, **kwargs):
    # we've loaded the images for you
    style_img = load_image(style_img_path)
    content_img = load_image(content_img_path)
    # output_path = "./images/output/content"

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
    if kwargs.get("reconstruct", False):
        print("Performing Image Reconstruction from white noise initialization")
        # input_img = random noise of the size of content_img on the correct device
        input_img = torch.rand_like(content_img, requires_grad=True, device=device)
        # output = reconstruct the image from the noise
        output = run_optimization(cnn, content_img, style_img, input_img, use_style=False, **kwargs)

        plt.figure()
        imshow(output, title='Reconstructed Image', save=True)

    # texture synthesis
    if kwargs.get("texture", False):
        print("Performing Texture Synthesis from white noise initialization")
        # input_img = random noise of the size of content_img on the correct device
        input_img = torch.rand_like(content_img, requires_grad=True, device=device)
        # output = synthesize a texture like style_image
        output = run_optimization(cnn, content_img, style_img, input_img, use_content=False, **kwargs)

        plt.figure()
        imshow(output, title='Synthesized Texture', save=True)

    # style transfer
    if kwargs.get("style_transfer", False):
        # # input_img = random noise of the size of content_img on the correct device
        # input_img = torch.rand_like(content_img, requires_grad=True, device=device)
        # # output = transfer the style from the style_img to the content image
        # output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, **kwargs)

        # plt.figure()
        # imshow(output, title='Output_noise', save=True)

        print("Performing Style Transfer from content image initialization")
        content_img: torch.Tensor
        input_img = content_img.clone().to(device).requires_grad_(True)
        # output = transfer the style from the style_img to the content image
        output = run_optimization(cnn, content_img, style_img, input_img, **kwargs)

        plt.figure()
        imshow(output, title='Output_content', save=True)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('style_image', type=str)
    parser.add_argument('content_image', type=str)
    parser.add_argument('--num_steps', type=int)
    parser.add_argument('--content_layer', type=int)
    parser.add_argument('--content_weight', type=float, default=1)
    parser.add_argument('--style_weight', type=float, default=1_000_000)
    parser.add_argument('--content_layers', help='delimited list input of content loss layer positions', 
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--style_layers', help='delimited list input of style loss layer positions', 
                        type=lambda s: [item for item in s.split(',')])
    
    parser.add_argument('--reconstruct', action=argparse.BooleanOptionalAction)
    parser.add_argument('--texture', action=argparse.BooleanOptionalAction)
    parser.add_argument('--style_transfer', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    main(args.style_image, args.content_image, **vars(args))
