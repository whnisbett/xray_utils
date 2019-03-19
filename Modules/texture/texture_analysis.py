import numpy as np
import imageproc_utils as ipu
import data_utils as du
import h5py
import glob
import os
from fuzzywuzzy import process
import copy

# def add_datum(datafile,texture_datum): have this function in an "modify_texturedata module/class". That class will be
# responsible for modifying the datafile (adding stuff, getting rid of stuff, searching for stuff, etc.)
# is there a way to check if datafile is writable?


# need a
def get_hdf5_file(dir):
    os.chdir(dir)
    if len(glob.glob("*.hdf5")) != 0:
        print("HDF5 Files in Directory: " + glob.glob("*.hdf5"))
        use_existing_files = input("Would you like to use one of these files?")
        if use_existing_files.lower() in ['yes', 'y', 'ye']:
            selected_file = input("Which file would you like to use?")
            hdf5_filename, _ = process.extractOne(selected_file, glob.glob("*.hdf5"))
        else:
            hdf5_filename = input('What would you like to name the new hdf5 file?')
    else:
        hdf5_filename = input('What would you like to name the new hdf5 file?')
    f = h5py.File(hdf5_filename, 'a')

    return f

def is_file_empty(file):
    return len(ls_group_hdf5(file)) == 0

def add_new_deepest_level(file,subgroups):

    deepest_parent = cd_deepest_parent(file)
    if is_file_empty(file):
        for subgroup in subgroups:
            deepest_parent.create_group(subgroup)
        return
    # IF CHILDREN EXIST, VISIT EACH ONE AND CREATE EACH SUBGROUP

    for child in deepest_parent:
        child_group = cd_group_hdf5(deepest_parent,child)
        for subgroup in subgroups:
            child_group.create_group(subgroup)

def cd_deepest_parent(file):
    # ASSUMING COMPLETELY SYMMETRIC TREE
    curr_group = file
    subgroups = ls_group_hdf5(curr_group)

    while len(subgroups) != 0:

        curr_group = cd_group_hdf5(curr_group,subgroups[0])
        subgroups = ls_group_hdf5(curr_group)

    deepest_parent = curr_group.parent
    return deepest_parent


def init_groups_hdf5(file,groups_dict):
    # GROUP DICTS: {'PD': [25,50,75],'Span': [15,30,45,60,75,90], ... }

    if len(ls_group_hdf5(file)) != 0:
        raise Exception('HDF5 file provided is not empty. Please provide an empty file for initialization.')
    for group in groups_dict:
        subgroups = groups_dict[group]
        add_new_deepest_level(file,subgroups)


def cd_group_hdf5(working_group,group_name):
    # NAVIGATE FROM WORKING_GROUP TO GROUP_NAME IF IT EXISTS, OTHERWISE CREATE AND RETURN IT
    try:
        group = working_group[group_name]
    except KeyError:
        group = working_group.create_group()
    return group

def get_parent(working_group):
    return working_group.parent

def ls_group_hdf5(group):
    return list(group.keys())

def add_datum_to_hdf5(hdf5_file,texture_datum):
    base_path = texture_datum.base_group_path
    base_group = cd_group_hdf5(hdf5_file,base_path)


    return
def hdf5_dset_to_datum(dset):
    return

def analyze_texture(dir,roi_coord):
    # NEED TO WORK ON A COUPLE OF FUNCTIONS: CHECK_FOR_HDF5_FILE, INITIALIZE_HDF5_GROUPS, GET_GROUPS_FROM_FILENAME <--
    # OUTPUT OF THIS FUNCTION SHOULD BE A DICTIONARY SO YOU CAN DO SOMETHING LIKE:
    # GET GROUPS, THEN COMPILE IT INTO A FULL GROUP PATH, STORE THIS AS AN ATTRIBUTE OF THE TEXTURE DATUM,
    # THEN ACCESS THAT PATH IN THE HDF5 AND STORE THE TEXTURE DATUM THERE
    f = get_hdf5_file(dir)
    # ---------------------------------
    for filename in os.listdir(dir):
        img = du.open_img(filename)
        if img == None:
            continue

        texture_datum = TextureDatum(img,roi_coord,filename = filename)
        try:
            dset = f[texture_datum.base_group_path].create_dataset("data",(1,1),dtype = 'O', maxshape = (None,1))
            # NEED TO THINK ABOUT THIS MORE. HOW CAN I STORE THIS OBJECT WELL...
        except RuntimeError:
            # APPEND THE DATA TO THE DSET
            raise



    return


class TextureDatum:
    def __init__(self, img, roi_coord, filename = None, n_grey = 256, quant_method = 'uniform'):
        self.img = img
        self.n_grey = n_grey
        self.quant_method = quant_method
        self.filename = filename
        self.parameter_groups = None # IMPLEMENT A FUNCTION TO GET THE PARAMETER GROUPS FROM THE FILENAME
        self.base_group_path = None # IMPLEMENT A FUNCTION TO GET THE CORRECT GROUP PATH FOR THE DATA FROM THE PARAMETER
        # GROUPS

        self.quantized_img = self.quantize_img(
            quantization_range = [np.percentile(self.img, 3), np.percentile(self.img, 97)], n_grey = self.n_grey,
            method = self.quant_method)
        self.rois = [ipu.select_roi(self.quantized_img,coord) for coord in roi_coord]
        self.glcms = [GLCM(roi, self.n_grey, d = 1, angle = 45) for roi in self.rois]
        # self.ngtdm = NGTDM(self.img, n = 1)
        # self.rlm = RLM(self.img, angle = 45)

    def quantize_img(self, quantization_range, n_grey = 256, method = 'uniform'):
        if method == 'uniform':
            quant_min = quantization_range[0]
            quant_max = quantization_range[1]
            relative_bins = (self.img - quant_min) / (quant_max - quant_min)
            print(quant_min)
            print(quant_max)
            print(relative_bins)
            # FIND ALL VALUES THAT ARE LESS THAN THE MIN THRESHOLD
            nonzero_values = relative_bins >= 0
            nonzero_values = nonzero_values.astype(np.int)
            print(nonzero_values)
            nonzero_img = np.floor(relative_bins * nonzero_values * (n_grey - 2))
            print(nonzero_img)
            max_values = nonzero_img >= (n_grey - 2)
            quantized_img = np.ma.array(nonzero_img, mask = max_values)
            print(quantized_img)
            quantized_img = np.ma.filled(quantized_img, fill_value = n_grey - 1)
            print(quantized_img)
            return quantized_img.astype(np.int)


class GLCM:
    def __init__(self, img, n_grey, d, angle):
        self.DEG2RAD = np.pi / 180
        self.img = img
        self.n_grey = n_grey
        self.d = d
        self.angle = angle

        i_array = np.linspace(0, self.n_grey - 1, num = self.n_grey)
        j_array = i_array
        j_grid, i_grid = np.meshgrid(i_array, j_array)

        self.glcm = self.get_glcm(self.img, self.n_grey, self.d, self.angle)

        self.p_x = np.sum(self.glcm, axis = 1)  # PROBABILITY OF GREY LEVEL I BEING FOCUS PIXEL
        self.p_y = np.sum(self.glcm, axis = 0)  # PROBABILITY OF GREY LEVEL J BEING THE COMPARISON PIXEL
        self.mean_i = np.sum(i_array*self.p_x)  # MEAN FOCUS PIXEL GREY LEVEL
        self.mean_j = np.sum(j_array*self.p_y)  # MEAN COMPARISON PIXEL GREY LEVEL
        self.std_i = np.sqrt(np.sum((i_array-self.mean_i)**2*self.p_x))  # STANDARD DEVIATION OF FOCUS PIXEL GREY LEVEL
        # PROBABILITY
        # DISTRIBUTION
        self.std_j = np.sqrt(np.sum((j_array-self.mean_j)**2*self.p_y))  # STANDARD DEVIATION OF COMPARISON PIXEL GREY
        # LEVEL
        # PROBABILITY
        # DISTRIBUTION

        self.features = {'entropy': self.entropy(), 'energy': self.energy(), 'inverse difference moment':
            self.inverse_difference_moment(), 'correlation': self.correlation()}  #
        ...

    def get_glcm(self, img, n_grey, d, angle):
        # INITIALIZE GLCM AND GET COMPARISON VECTOR
        glcm = np.zeros((n_grey, n_grey))
        n_row, n_col = img.shape
        [dr, dc] = self.find_comparison_vector(d, angle)
        # DECIDE WHETHER TO START AWAY FROM LEFT EDGE OR END BEFORE RIGHT EDGE (COMPARISON VECTOR CAN'T GO OUTSIDE IMG)
        # UNLESS DR IS 0, ALWAYS NEED TO START DOWN IN THE Y DIRECTION

        if np.sign(dc) == -1:
            # IF DC IS NEGATIVE, NEED TO START INWARD
            for r in range(np.abs(dr), n_row):
                for c in range(np.abs(dc), n_col):
                    focus_gl = img[r, c]
                    comp_gl = img[r + dr, c + dc]
                    glcm[focus_gl, comp_gl] += 1
        else:
            # IF DC IS POSITIVE, NEED TO END EARLY (OR IF DC IS ZERO JUST USE THE WHOLE IMG)
            for r in range(np.abs(dr), n_row):
                for c in range(n_col - dc):
                    focus_gl = img[r, c]
                    comp_gl = img[r + dr, c + dc]
                    glcm[focus_gl][comp_gl] += 1

        # NORMALIZE AND RETURN GLCM
        return glcm / np.sum(glcm)

    def find_comparison_vector(self, d, angle):
        # find the vector pointing from the focus pixel to the comparison pixel for calculating the GLCM
        # Angle should be in set [0,180)
        tan = np.tan(angle * self.DEG2RAD)
        # One component of the vector must be +/- d and the other component must be smaller or equal in magnitude
        # than d. We can figure out which component is d by looking at tan(ø) and then deduce the other component
        if tan == 0:
            dc = d
            dr = 0
        elif np.abs(tan) > 1000:
            dc = 0
            dr = -d
        elif np.abs(tan) > 1:
            dr = -d
            dc = np.round(np.abs(dr / tan))
        elif np.abs(tan) < 1:
            dc = d
            dr = -np.round(np.abs(dc * tan))
        # elif np.abs(tan) == 1:
        else:
            dc = d
            dr = -d

        if np.sign(tan) == -1:
            dc = -dc

        return np.int(dr), np.int(dc)

    def inverse_difference_moment(self):
        i_array = np.linspace(0, self.n_grey - 1, num = self.n_grey)
        j_array = i_array

        gl_diff_mat = np.subtract.outer(i_array, j_array)
        inverse_difference_mat = (1 / (1 + gl_diff_mat**2)) * self.glcm
        inverse_difference_moment = np.sum(inverse_difference_mat)

        return inverse_difference_moment

    def correlation(self):
        i_array = np.linspace(0, self.n_grey - 1, num = self.n_grey)
        j_array = i_array
        j_grid, i_grid = np.meshgrid(i_array, j_array)

        cov_ij = np.sum(i_grid * j_grid * self.glcm) - self.mean_i * self.mean_j
        correlation = cov_ij / (self.std_i * self.std_j)

        return correlation

    def entropy(self):
        glcm_eps = np.log2(self.glcm + 1E-10)
        ent_mat = glcm_eps * self.glcm
        entropy = -np.sum(ent_mat)

        return entropy

    def energy(self):

        energy = np.sum(self.glcm * self.glcm)

        return energy
    # ...


class NGTDM:
    def __init__(self, img, n):
        hood_xgrid = np.linspace(-n, n, 2 * n + 1)
        hood_ygrid = np.linspace(-n, n, 2 * n + 1)

        hood_grid = np.meshgrid(hood_xgrid, hood_ygrid)

        return


class RLM:
    def __init__(self, img, angle):
        return
