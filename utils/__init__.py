
EDSR = "EDSR"
RDN = "RDN"
RCAN = "RCAN"
SwinIR = "SwinIR"
NLSN = "NLSN"
architecture_options = [EDSR, RDN, RCAN, SwinIR, NLSN]
L1 = "L1"
VGG = "VGG"
ResNet = "ResNet"
loss_options = [L1, VGG, ResNet]
flickr2k = "flickr2k"
div2k = "div2k"
quarter_div2k = "quarter_div2k"
quarter_flickr2k = "quarter_flickr2k"
dataset_options = [flickr2k, div2k, quarter_div2k, quarter_flickr2k]
scale_options = [2, 4]
seed_options = [1, 2, 3]

def get_sisr_model_name(arch,scale,loss,dataset,seed):
	raise NotImplementedError()

def get_sisr_model_names(
	arch=architecture_options,
	scale=scale_options,
	loss=loss_options,
	dataset=dataset_options,
	seed=seed_options):
	   
	all_combos = list(
        product(architecture, scale, loss, dataset, seed)
    )
    all_combos = [
        (arch, scale, loss, dataset, seed)
        for arch, scale, loss, dataset, seed in all_combos
        if seed == 1 or dataset == div2k
    ]

    return [get_sisr_model_name(*combo) for combo in all_combos]



