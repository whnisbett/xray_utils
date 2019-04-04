from texture import texture_analysis as ta

glcm_features = {'entropy': [], 'energy': [], 'inverse difference moment': []
    , 'correlation': []}
ngtdm_features = {'complexity': [], 'busyness' : [], 'coarseness':[], 'contrast': []}
filenames = []
for i,var in enumerate(list(locals().values())):
    if isinstance(var, ta.TextureDatum):
        filenames.append(list(locals().keys())[i])
        for i in range(len(var.rois)):
            curr_glcm = var.glcms[i].features
            curr_ngtdm = var.ngtdms[i].features
            for feature in curr_glcm:

                glcm_features[feature].append(curr_glcm[feature])
            for feature in curr_ngtdm:

                ngtdm_features[feature].append(curr_ngtdm[feature])
