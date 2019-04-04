import data_utils as du
import plotting_utils as pu
from texture import texture_analysis as ta

ff_dir = '/home/billy/Research/Projects/08_30_airGap_thickerSlabs/1000umCdTe/40kvp/ff/'
img_dir = '/home/billy/Research/Projects/08_30_airGap_thickerSlabs/1000umCdTe/40kvp/pos_4cm/thk_51mm/'

ff_data = du.import_data(ff_dir)
img_data = du.import_data(img_dir)

img_data.ff_corr(ff_data)



# iterate through the data and pair up ff data with image data

