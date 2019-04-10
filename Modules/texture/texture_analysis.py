import os

import numpy as np

import data_utils as du
from texture import hdf5_utils as hu


# def add_datum(datafile,texture_datum): have this function in an "modify_texturedata
# module/class". That class will be
# responsible for modifying the datafile (adding stuff, getting rid of stuff, searching for
# stuff, etc.)
# is there a way to check if datafile is writable?


# need a

def analyze_texture(dir, roi_coord):
    # NEED TO WORK ON A COUPLE OF FUNCTIONS: CHECK_FOR_HDF5_FILE, INITIALIZE_HDF5_GROUPS,
    # GET_GROUPS_FROM_FILENAME <--
    # OUTPUT OF THIS FUNCTION SHOULD BE A DICTIONARY SO YOU CAN DO SOMETHING LIKE:
    # GET GROUPS, THEN COMPILE IT INTO A FULL GROUP PATH, STORE THIS AS AN ATTRIBUTE OF THE
    # TEXTURE DATUM,
    # THEN ACCESS THAT PATH IN THE HDF5 AND STORE THE TEXTURE DATUM THERE
    f = hu.get_hdf5_file(dir)
    # ---------------------------------
    for filename in os.listdir(dir):
        img = du.open_img(filename)
        if img == None:
            continue

        texture_datum = TextureDatum(img, roi_coord, filename=filename)
        try:
            dset = f[texture_datum.base_group_path].create_dataset("data", (1, 1), dtype='O',
                                                                   maxshape=(None, 1))
            # NEED TO THINK ABOUT THIS MORE. HOW CAN I STORE THIS OBJECT WELL...
        except RuntimeError:
            # APPEND THE DATA TO THE DSET
            raise

    return


class TextureDatum:
    def __init__(self, rois, img=None, roi_coord=None, filename=None, n_grey=256,
                 quant_method='uniform'):
        # initialize variables
        self.img = img
        self.n_grey = n_grey
        self.quant_method = quant_method
        self.filename = filename
        self.rois = rois
        self.parameter_groups = None  # IMPLEMENT A FUNCTION TO GET THE PARAMETER GROUPS FROM THE
        # FILENAME
        self.base_group_path = None  # IMPLEMENT A FUNCTION TO GET THE CORRECT GROUP PATH FOR THE
        # DATA FROM THE PARAMETER
        # GROUPS

        # quantize the rois
        # self.quant_rois = [
        #     ipu.quantize_img(roi, quantization_range=[np.percentile(roi, 1), np.percentile(roi,
        #     99)],
        #                       n_grey=self.n_grey, method=self.quant_method) for roi in self.rois]
        # compute texture matrices and features for each roi

        self.glcms = [GLCM(roi, self.n_grey, d=1, angle=45) for roi in self.rois]
        self.ngtdms = [NGTDM(roi, n_grey=256, d=1) for roi in self.rois]
        self.compute_feature_histogram()
        # self.rlm = RLM(self.img, angle = 45)

    def compute_feature_histogram(self):
        feature_list = {'entropy': [], 'energy': [], 'inverse difference moment':
            [], 'correlation': [], 'busyness': [], 'complexity': [],'coarseness': [], 'contrast': []}
        for i in range(len(self.rois)):
            glcm_feats = self.glcms[i].features
            ngtdm_feats = self.ngtdms[i].features

            for feature in glcm_feats.keys():
                feature_list[feature].append(glcm_feats[feature])
            for feature in ngtdm_feats.keys():
                feature_list[feature].append(ngtdm_feats[feature])

        self.feature_list = feature_list

class GLCM:
    def __init__(self, img, n_grey, d, angle):
        # initialize variables
        self.DEG2RAD = np.pi / 180
        self.img = img
        self.n_grey = n_grey
        self.d = d
        self.angle = angle

        # compute GLCM for the image
        self.glcm = self.get_glcm(self.img, self.n_grey, self.d, self.angle)

        # create two grey level vectors i and j with length n_grey (e.g. 256)
        i_array = np.linspace(0, self.n_grey - 1, num=self.n_grey)
        j_array = i_array

        # calculate normalized probabilities, means, and standard deviations for focus and
        # comparison pixel gray levels
        self.p_x = np.sum(self.glcm, axis=1)  # PROBABILITY OF GREY LEVEL I BEING FOCUS PIXEL
        self.p_y = np.sum(self.glcm,
                          axis=0)  # PROBABILITY OF GREY LEVEL J BEING THE COMPARISON PIXEL
        self.mean_i = np.sum(i_array * self.p_x)  # MEAN FOCUS PIXEL GREY LEVEL
        self.mean_j = np.sum(j_array * self.p_y)  # MEAN COMPARISON PIXEL GREY LEVEL
        self.std_i = np.sqrt(np.sum(
            (i_array - self.mean_i) ** 2 * self.p_x)
        )  # STANDARD DEVIATION OF FOCUS PIXEL GREY LEVEL PROBABILITY DISTRIBUTION
        self.std_j = np.sqrt(np.sum(
            (j_array - self.mean_j) ** 2 * self.p_y)
        )  # STANDARD DEVIATION OF COMPARISON PIXEL GREY LEVEL PROBABILITY DISTRIBUTION
        # compute all GLCM features
        self.features = {'entropy': self.entropy(), 'energy': self.energy(),
                         'inverse difference moment':
                             self.inverse_difference_moment(), 'correlation': self.correlation()}

    def get_glcm(self, img, n_grey, d, angle):
        # INITIALIZE GLCM AND GET COMPARISON VECTOR
        glcm = np.zeros((n_grey, n_grey))
        n_row, n_col = img.shape
        [dr, dc] = self.find_comparison_vector(d, angle)
        # DECIDE WHETHER TO START AWAY FROM LEFT EDGE OR END BEFORE RIGHT EDGE (COMPARISON VECTOR
        # CAN'T GO OUTSIDE IMG)
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
        # find the vector pointing from the focus pixel to the comparison pixel for calculating
        # the GLCM
        # Angle should be in set [0,180)
        tan = np.tan(angle * self.DEG2RAD)
        # One component of the vector must be +/- d and the other component must be smaller or
        # equal in magnitude
        # than d. We can figure out which component is d by looking at tan(ø) and then deduce the
        # other component
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
        i_array = np.linspace(0, self.n_grey - 1, num=self.n_grey)
        j_array = i_array

        gl_diff_mat = np.subtract.outer(i_array, j_array)
        inverse_difference_mat = (1 / (1 + gl_diff_mat ** 2)) * self.glcm
        inverse_difference_moment = np.sum(inverse_difference_mat)

        return inverse_difference_moment

    def correlation(self):
        i_array = np.linspace(0, self.n_grey - 1, num=self.n_grey)
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


class NGTDM:
    def __init__(self, img, n_grey, d):
        self.d = d
        self.img = img
        self.n_grey = n_grey
        self.n_row, self.n_col = np.shape(self.img)
        self.hood_size = (2 * self.d + 1) ** 2 - 1
        self.effective_img_size = (self.n_row - 2 * self.d) * (self.n_col - 2 * self.d)

        self.ngtdm = np.zeros(self.n_grey)
        self.get_ngtdm()
        self.p_i = self.get_probability_vector()
        self.features = {'busyness': self.busyness(), 'complexity': self.complexity(),
                         'coarseness': self.coarseness(

                         ), 'contrast': self.contrast()}

    def get_probability_vector(self):
        p_i = np.zeros(self.n_grey)

        for i in range(self.n_grey):
            p_i[i] = np.sum(self.img == i)
        p_i = p_i / (self.n_row * self.n_col)

        return p_i

    def get_ngtdm(self):

        for r in range(self.d, self.n_row - self.d):
            for c in range(self.d, self.n_col - self.d):
                curr_gl = self.img[r, c]
                hood = self.get_neighborhood([r, c])
                hood_avg = (np.sum(hood) - curr_gl) / self.hood_size
                gtd = np.abs(curr_gl - hood_avg)
                self.ngtdm[curr_gl] += gtd

    def get_neighborhood(self, pixel_coord):

        try:
            hood = self.img[(pixel_coord[0] - self.d):(pixel_coord[0] + self.d + 1),
                   (pixel_coord[1] - self.d):(pixel_coord[1] + self.d + 1)]
        except IndexError:
            raise Exception('Neighborhood extends out of image bounds')

        return hood

    def busyness(self):

        existing_gl = np.nonzero(self.p_i)
        non0_p_i = self.p_i[existing_gl]
        non0_gl = np.arange(self.n_grey)[existing_gl]

        weighted_i = non0_p_i * non0_gl
        weighted_ngtdm = self.ngtdm[existing_gl] * non0_p_i

        denom = np.sum(np.abs(np.subtract.outer(weighted_i, weighted_i)))
        num = np.sum(weighted_ngtdm)
        busyness = num / denom
        return busyness

    def complexity(self):
        existing_gl = np.nonzero(self.p_i)
        non0_p_i = self.p_i[existing_gl]
        non0_gl = np.arange(self.n_grey)[existing_gl]

        term1_num = np.abs(np.subtract.outer(non0_gl, non0_gl))
        term1_denom = np.add.outer(non0_p_i, non0_p_i) * self.effective_img_size
        term2 = np.add.outer(non0_p_i * self.ngtdm[existing_gl], non0_p_i * self.ngtdm[existing_gl])
        complexity = np.sum((term1_num / term1_denom) * term2)

        return complexity

    def contrast(self):
        existing_gl = np.nonzero(self.p_i)
        non0_p_i = self.p_i[existing_gl]
        non0_gl = np.arange(self.n_grey)[existing_gl]
        n_non0_gl = len(non0_gl)

        denom = np.sum(self.ngtdm) / self.effective_img_size
        num_1 = 1 / (n_non0_gl * (n_non0_gl - 1))
        num_2 = np.multiply.outer(non0_p_i, non0_p_i)
        num_3 = np.subtract.outer(non0_gl, non0_gl) ** 2
        num = num_1 * np.sum(num_2 * num_3)

        contrast = num / denom
        return contrast

    def coarseness(self):

        existing_gl = np.nonzero(self.p_i)
        non0_p_i = self.p_i[existing_gl]

        weighted_ngtdm = self.ngtdm[existing_gl] * non0_p_i

        coarseness = 1 / (np.sum(weighted_ngtdm))

        return coarseness


class RLM:
    def __init__(self, img, angle):
        return
