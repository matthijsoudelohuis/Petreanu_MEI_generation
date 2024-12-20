
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

# Set working directory to root of repo
current_path = os.getcwd()

filename        = 'meis_top40.pth'
fullfile        = os.path.join(current_path,'runs','with_variability',filename)

session_id      = 'LPE10884_2023_10_20'

outdir          = os.path.join('E:\\Bonsai\\lab-leopoldo-solene-vr\\workflows\\MEIs\\',session_id)
os.makedirs(f'{outdir}', exist_ok=True)

meis            = torch.load(fullfile)
nmeis           = len(meis)
meidim          = np.array(meis[0].shape[2:])

# load cell_ids:
# cell_ids    = np.load....
cell_ids = [f'{session_id}_{imei:03d}' for imei in range(nmeis)] #give some temp cell id

for imei,mei_out in enumerate(meis):
    mei_out = np.array(mei_out[0, 0, ...])
    mei_out = (mei_out + 1) / 2
    mei_out = np.concatenate((np.full(meidim,0.5),mei_out), axis=1) #add left part of the screen
    mei_out = (mei_out * 255).astype(np.uint8)
    # np.save(os.path.join(outdir,'%d.jpg' % imei),mei_out)
    img = Image.fromarray(mei_out)
    img.save(os.path.join(outdir,'%s.jpg' % cell_ids[imei]), format='JPEG')



