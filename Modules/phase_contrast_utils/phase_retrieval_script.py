import numpy as np
import data_utils as du
import stat_utils as su
ff_data = du.import_data(directory = '/Users/billy/Downloads/PCI/data/09-24-2018/ff',
                      data_type = 'spectral')
data = du.import_data(directory = '/Users/billy/Downloads/PCI/data/09-24-2018/rod_b/bulk_0mm',
                      data_type = 'spectral')
from matplotlib import pyplot as plt
import imageproc_utils as ipu
data.ff_corr(ff_data)
import plotting_utils as pu
experiment = du.Experiment(1,3,detector = du.Detector('MediPix3RX',55,(256,256)),kvp = 50, current = 140,spot_size =
'small')
experiment.add_data(data)
from phase_contrast_utils import spectral_phase_retrieval as spr
maps = spr.main(experiment)

spatial_maps = np.fft.ifft()