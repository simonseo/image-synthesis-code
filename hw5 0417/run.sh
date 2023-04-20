# python main.py --model vanilla --mode project --latent z
# python main.py --model stylegan --mode project --latent z
# python main.py --model stylegan --mode project --latent w
# python main.py --model stylegan --mode project --latent w+


# python main.py --model vanilla --mode project --latent z --l1_wgt 10 --l2_wgt 3 --perc_wgt 0
# python main.py --model stylegan --mode project --latent z --l1_wgt 10 --l2_wgt 3 --perc_wgt 0
# python main.py --model stylegan --mode project --latent w --l1_wgt 10 --l2_wgt 3 --perc_wgt 0
# python main.py --model stylegan --mode project --latent w+ --l1_wgt 10 --l2_wgt 3 --perc_wgt 0


# python main.py --model vanilla --mode project --latent z --l1_wgt 10 --l2_wgt 10 --perc_wgt 0.1
# python main.py --model stylegan --mode project --latent z --l1_wgt 10 --l2_wgt 10 --perc_wgt 0.1
# python main.py --model stylegan --mode project --latent w --l1_wgt 10 --l2_wgt 10 --perc_wgt 0.1
# python main.py --model stylegan --mode project --latent w+ --l1_wgt 10 --l2_wgt 10 --perc_wgt 0.1


# python main.py --model stylegan --mode interpolate --latent w+ --l1_wgt 10 --l2_wgt 10 --perc_wgt 0.1 --resolution 256
# python main.py --model vanilla --mode interpolate --latent z --l1_wgt 10 --l2_wgt 10 --perc_wgt 0.1 --resolution 64


# python main.py --model stylegan --mode draw --latent w+ --l1_wgt 10 --perc_wgt 0.1 --resolution 64 --input 'data/sketch/cat512.png'
# python main.py --model vanilla --mode draw --latent z --l1_wgt 10 --perc_wgt 0.01 --resolution 64 --input 'data/sketch/cat512.png'
# python main.py --model stylegan --mode draw --latent w+ --l1_wgt 10 --perc_wgt 0.01 --resolution 64 --input 'data/sketch/boy.png'
python main.py --model stylegan --mode draw --latent w+ --l1_wgt 10 --perc_wgt 0.5 --input 'data/sketch/*.png'