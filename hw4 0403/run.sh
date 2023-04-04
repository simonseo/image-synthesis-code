# reconstruct
# python run.py images/style/frida_kahlo.jpeg images/content/phipps.jpeg --reconstruct --num_steps 5 --content_layer 1
# python run.py images/style/frida_kahlo.jpeg images/content/phipps.jpeg --reconstruct --num_steps 5 --content_layer 2
# python run.py images/style/frida_kahlo.jpeg images/content/phipps.jpeg --reconstruct --num_steps 5 --content_layer 3
# python run.py images/style/frida_kahlo.jpeg images/content/phipps.jpeg --reconstruct --num_steps 5 --content_layer 4
# python run.py images/style/frida_kahlo.jpeg images/content/phipps.jpeg --reconstruct --num_steps 5 --content_layer 5
# python run.py images/style/frida_kahlo.jpeg images/content/phipps.jpeg --reconstruct --num_steps 5 --content_layer 6
# python run.py images/style/frida_kahlo.jpeg images/content/phipps.jpeg --reconstruct --num_steps 5 --content_layer 7
# python run.py images/style/frida_kahlo.jpeg images/content/phipps.jpeg --reconstruct --num_steps 5 --content_layer 8
# python run.py images/style/frida_kahlo.jpeg images/content/phipps.jpeg --reconstruct --num_steps 5 --content_layer 9

# texture synthesis
# python run.py images/style/frida_kahlo.jpeg images/content/phipps.jpeg --num_steps 300 --texture
# python run.py images/style/escher_sphere.jpeg images/content/phipps.jpeg --num_steps 30 --texture
# python run.py images/style/picasso.jpg images/content/phipps.jpeg --num_steps 30 --texture
# python run.py images/style/starry_night.jpeg images/content/phipps.jpeg --num_steps 30 --texture
# python run.py images/style/the_scream.jpeg images/content/phipps.jpeg --num_steps 30 --texture

# python run.py images/style/frida_kahlo.jpeg images/style/frida_kahlo.jpeg --num_steps 10 --texture --style_layers "conv_1,conv_2,conv_3,conv_4,conv_5,conv_6,conv_7,conv_8,conv_9,conv_10"
# python run.py images/style/frida_kahlo.jpeg images/style/frida_kahlo.jpeg --num_steps 10 --texture --style_layers "conv_1,conv_2,conv_3,conv_4,conv_5,conv_6,conv_7,conv_8,conv_9,conv_10,conv_11,conv_12,conv_13,conv_14,conv_15,conv_16"
# python run.py images/style/frida_kahlo.jpeg images/style/frida_kahlo.jpeg --num_steps 10 --texture --style_layers "conv_2,conv_4,conv_8,conv_12,conv_16"
# python run.py images/style/frida_kahlo.jpeg images/style/frida_kahlo.jpeg --num_steps 10 --texture --style_layers "conv_13,conv_14,conv_15,conv_16"
# python run.py images/style/frida_kahlo.jpeg images/style/frida_kahlo.jpeg --num_steps 10 --texture --style_layers "conv_1,conv_2,conv_3,conv_14,conv_15,conv_16"
# python run.py images/style/frida_kahlo.jpeg images/style/frida_kahlo.jpeg --num_steps 50 --texture --style_layers "conv_1,conv_2"
# python run.py images/style/frida_kahlo.jpeg images/style/frida_kahlo.jpeg --num_steps 10 --texture --style_layers "conv_3,conv_4"
# python run.py images/style/frida_kahlo.jpeg images/style/frida_kahlo.jpeg --num_steps 50 --texture --style_layers "conv_5,conv_6,conv_7,conv_8"
# python run.py images/style/frida_kahlo.jpeg images/style/frida_kahlo.jpeg --num_steps 50 --texture --style_layers "conv_9,conv_10,conv_11,conv_12"
# python run.py images/style/frida_kahlo.jpeg images/style/frida_kahlo.jpeg --num_steps 10 --texture --style_layers "conv_1,conv_5,conv_8"

# python run.py images/style/starry_night.jpeg images/style/starry_night.jpeg --num_steps 10 --texture --style_layers "conv_1,conv_5,conv_8"
# python run.py images/style/starry_night.jpeg images/style/starry_night.jpeg --num_steps 10 --texture --style_layers "conv_1,conv_2,conv_3,conv_4,conv_5,conv_6,conv_7,conv_8,conv_9,conv_10,conv_11,conv_12,conv_13,conv_14,conv_15,conv_16"
# python run.py images/style/starry_night.jpeg images/style/starry_night.jpeg --num_steps 10 --texture --style_layers "conv_4,conv_8,conv_12,conv_16"
# python run.py images/style/starry_night.jpeg images/style/starry_night.jpeg --num_steps 10 --texture --style_layers "conv_3,conv_4,conv_8,conv_12,conv_16" --style_weight 10
# python run.py images/style/starry_night.jpeg images/style/starry_night.jpeg --num_steps 10 --texture --style_layers "conv_3,conv_4,conv_8,conv_12,conv_16" --style_weight 30
# python run.py images/style/starry_night.jpeg images/style/starry_night.jpeg --num_steps 10 --texture --style_layers "conv_3,conv_4,conv_8,conv_12,conv_16" --style_weight 50
# python run.py images/style/starry_night.jpeg images/style/starry_night.jpeg --num_steps 10 --texture --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 1
# python run.py images/style/starry_night.jpeg images/style/starry_night.jpeg --num_steps 10 --texture --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 0.5
# python run.py images/style/starry_night.jpeg images/style/starry_night.jpeg --num_steps 10 --texture --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 0.5 --content_weight 0

# python run.py images/style/picasso.jpg images/style/picasso.jpg --num_steps 10 --texture --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 0.5 --content_weight 0
# python run.py images/style/frida_kahlo.jpeg images/style/frida_kahlo.jpeg --num_steps 10 --texture --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 0.5 --content_weight 0
# python run.py images/style/escher_sphere.jpeg images/style/the_scream.jpeg --num_steps 10 --texture --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 0.5 --content_weight 0
# python run.py images/style/the_scream.jpeg images/style/the_scream.jpeg --num_steps 10 --texture --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 0.5 --content_weight 0


# style transfer
# python run.py images/style/the_scream.jpeg images/content/dancing.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/the_scream.jpeg images/content/fallingwater.png --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/the_scream.jpeg images/content/phipps.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/the_scream.jpeg images/content/tubingen.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/the_scream.jpeg images/content/wally.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/starry_night.jpeg images/content/dancing.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/starry_night.jpeg images/content/fallingwater.png --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/starry_night.jpeg images/content/phipps.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/starry_night.jpeg images/content/tubingen.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/starry_night.jpeg images/content/wally.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/escher_sphere.jpeg images/content/dancing.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/escher_sphere.jpeg images/content/fallingwater.png --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/escher_sphere.jpeg images/content/phipps.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/escher_sphere.jpeg images/content/tubingen.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/escher_sphere.jpeg images/content/wally.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/picasso.jpg images/content/dancing.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/picasso.jpg images/content/fallingwater.png --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/picasso.jpg images/content/phipps.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/picasso.jpg images/content/tubingen.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/picasso.jpg images/content/wally.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/frida_kahlo.jpeg images/content/dancing.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/frida_kahlo.jpeg images/content/fallingwater.png --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/frida_kahlo.jpeg images/content/phipps.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/frida_kahlo.jpeg images/content/tubingen.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/frida_kahlo.jpeg images/content/wally.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/simonseo.jpeg images/content/dancing.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/simonseo.jpeg images/content/fallingwater.png --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/simonseo.jpeg images/content/phipps.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/simonseo.jpeg images/content/tubingen.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/simonseo.jpeg images/content/wally.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/simonseo.jpeg images/content/cmusnow.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0

# python run.py images/style/benthomas.jpeg images/content/dancing.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/benthomas.jpeg images/content/fallingwater.png --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/benthomas.jpeg images/content/phipps.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/benthomas.jpeg images/content/tubingen.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/benthomas.jpeg images/content/wally.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/benthomas.jpeg images/content/cmusnow.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/benthomas.jpeg images/content/cyclist.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0


# python run.py images/style/fireman.jpeg images/content/dancing.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/fireman.jpeg images/content/fallingwater.png --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/fireman.jpeg images/content/phipps.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/fireman.jpeg images/content/tubingen.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/fireman.jpeg images/content/wally.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/fireman.jpeg images/content/cmusnow.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0
# python run.py images/style/fireman.jpeg images/content/cyclist.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 10.0 --content_weight 1.0



# python run.py images/style/simonseo.jpeg images/content/cmusnow.jpeg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 1_000_000.0 --content_weight 1.0 --content_layer 9
# python run.py images/style/simonseo2.jpg images/content/dancing.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# python run.py images/style/harryseo.jpg images/content/dancing.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# python run.py images/style/harryseo2.jpg images/content/dancing.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# python run.py images/style/harryseo3.jpeg images/content/dancing.jpg --num_steps 10 --style_transfer --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"\

# python run.py images/style/simonseo.jpeg images/style/simonseo.jpeg --num_steps 10 --texture --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 0.5 --content_weight 0
# python run.py images/style/simonseo3.jpg images/style/simonseo.jpeg --num_steps 10 --texture --style_layers "conv_3,conv_4,conv_5,conv_6" --style_weight 0.5 --content_weight 0
# python run.py images/style/simonseo2.jpg images/style/simonseo.jpeg --num_steps 10 --texture --style_layers "conv_4,conv_8,conv_16" --style_weight 0.5 --content_weight 0
# python run.py images/style/simonseo2.jpg images/style/simonseo.jpeg --num_steps 10 --texture --style_layers "conv_8,conv_16" --style_weight 0.5 --content_weight 0
# python run.py images/style/simonseo2.jpg images/style/simonseo.jpeg --num_steps 10 --texture --style_layers "conv_16" --style_weight 0.5 --content_weight 0
# python run.py images/style/simonseo3.jpg images/style/simonseo.jpeg --num_steps 10 --texture --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 0.5 --content_weight 0
# python run.py images/style/harryseo.jpg images/style/harryseo.jpg --num_steps 10 --texture --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 0.5 --content_weight 0
# python run.py images/style/harryseo2.jpg images/style/harryseo2.jpg --num_steps 10 --texture --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 0.5 --content_weight 0
# python run.py images/style/harryseo3.jpeg images/style/harryseo3.jpeg --num_steps 10 --texture --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 0.5 --content_weight 0

# python run.py images/style/simonseo3.jpg images/style/fireman.jpeg --num_steps 10 --style_transfer --style_layers "conv_3,conv_4,conv_5,conv_6" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# python run.py images/style/simonseo3.jpg images/style/frida_kahlo.jpeg --num_steps 10 --style_transfer --style_layers "conv_3,conv_4,conv_5,conv_6" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# python run.py images/style/simonseo3.jpg images/style/starry_night.jpeg --num_steps 10 --style_transfer --style_layers "conv_3,conv_4,conv_5,conv_6" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# python run.py images/style/simonseo3.jpg images/content/fallingwater.png --num_steps 10 --style_transfer --style_layers "conv_3,conv_4,conv_5,conv_6" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# python run.py images/style/simonseo3.jpg images/content/phipps.jpeg --num_steps 10 --style_transfer --style_layers "conv_3,conv_4,conv_5,conv_6" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# python run.py images/style/simonseo3.jpg images/content/tubingen.jpeg --num_steps 10 --style_transfer --style_layers "conv_3,conv_4,conv_5,conv_6" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"

# python run.py images/style/fireman.jpeg images/style/fireman.jpeg --num_steps 10 --texture --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 0.5 --content_weight 0
# python run.py images/style/fireman.jpeg images/style/fireman.jpeg --num_steps 10 --texture --style_layers "conv_3,conv_4,conv_5,conv_6" --style_weight 0.5 --content_weight 0
# python run.py images/style/fireman.jpeg images/style/fireman.jpeg --num_steps 10 --texture --style_layers "conv_3,conv_2,conv_1" --style_weight 1000000 --content_weight 0
# python run.py images/style/fireman.jpeg images/content/dancing.jpg --num_steps 10 --style_transfer --style_layers "conv_3,conv_2,conv_1" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# python run.py images/style/fireman.jpeg images/content/cyclist.jpeg --num_steps 10 --style_transfer --style_layers "conv_3,conv_2,conv_1" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# pyth·on run.py images/style/fireman.jpeg images/style/simonseo.jpeg --num_steps 10 --style_transfer --style_layers "conv_3,conv_2,conv_1" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# python run.py images/style/fireman.jpeg images/style/frida_kahlo.jpeg --num_steps 10 --style_transfer --style_layers "conv_3,conv_2,conv_1" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# python run.py images/style/fireman.jpeg images/style/starry_night.jpeg --num_steps 10 --style_transfer --style_layers "conv_3,conv_2,conv_1" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# python run.py images/style/fireman.jpeg images/content/tubingen.jpeg --num_steps 10 --style_transfer --style_layers "conv_3,conv_2,conv_1" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# python run.py images/style/fireman.jpeg images/content/phipps.jpeg --num_steps 10 --style_transfer --style_layers "conv_3,conv_2,conv_1" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# python run.py images/style/fireman.jpeg images/content/cmusnow.jpeg --num_steps 10 --style_transfer --style_layers "conv_3,conv_2,conv_1" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"
# python run.py images/style/fireman.jpeg images/content/wally.jpg --num_steps 10 --style_transfer --style_layers "conv_3,conv_2,conv_1" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_9"


# python run.py images/style/foggypit.jpeg images/style/foggypit.jpeg --num_steps 10 --texture --style_layers "conv_2,conv_4,conv_8,conv_16" --style_weight 0.5 --content_weight 0
# python run.py images/style/foggypit.jpeg images/style/foggypit.jpeg --num_steps 10 --texture --style_layers "conv_3,conv_4,conv_5,conv_6" --style_weight 0.5 --content_weight 0
# python run.py images/style/foggypit.jpeg images/style/foggypit.jpeg --num_steps 10 --texture --style_layers "conv_3,conv_2,conv_1" --style_weight 1000000 --content_weight 0

python run.py images/style/foggypit.jpeg images/content/dancing.jpg --num_steps 10 --style_transfer --style_layers "conv_1,conv_3,conv_4,conv_5,conv_6" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_2"
python run.py images/style/foggypit.jpeg images/content/cyclist.jpeg --num_steps 10 --style_transfer --style_layers "conv_1,conv_3,conv_4,conv_5,conv_6" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_2"
python run.py images/style/foggypit.jpeg images/style/simonseo.jpeg --num_steps 10 --style_transfer --style_layers "conv_1,conv_3,conv_4,conv_5,conv_6" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_2"
python run.py images/style/foggypit.jpeg images/style/frida_kahlo.jpeg --num_steps 10 --style_transfer --style_layers "conv_1,conv_3,conv_4,conv_5,conv_6" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_2"
python run.py images/style/foggypit.jpeg images/style/starry_night.jpeg --num_steps 10 --style_transfer --style_layers "conv_1,conv_3,conv_4,conv_5,conv_6" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_2"
python run.py images/style/foggypit.jpeg images/content/tubingen.jpeg --num_steps 10 --style_transfer --style_layers "conv_1,conv_3,conv_4,conv_5,conv_6" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_2"
python run.py images/style/foggypit.jpeg images/content/phipps.jpeg --num_steps 10 --style_transfer --style_layers "conv_1,conv_3,conv_4,conv_5,conv_6" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_2"
python run.py images/style/foggypit.jpeg images/content/cmusnow.jpeg --num_steps 10 --style_transfer --style_layers "conv_1,conv_3,conv_4,conv_5,conv_6" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_2"
python run.py images/style/foggypit.jpeg images/content/wally.jpg --num_steps 10 --style_transfer --style_layers "conv_1,conv_3,conv_4,conv_5,conv_6" --style_weight 1_000_000.0 --content_weight 1.0 --content_layers "conv_2"
