import os
import glob
import h5py
from fuzzywuzzy import process

def get_hdf5_file(dir):
    # GO TO INPUT DIRECTORY
    os.chdir(dir)

    if len(glob.glob("*.hdf5")) != 0:
        # PRINT ALL FILE NAMES AND PROMPT TO PICK ONE
        print("HDF5 Files in Directory: " + glob.glob("*.hdf5"))
        use_existing_files = input("Would you like to use one of these files?")
        if use_existing_files.lower() in ['yes', 'y', 'ye']:
            # IF USER WANTS TO PICK ONE, FIND CLOSEST MATCH TO THEIR INPUT
            selected_file = input("Which file would you like to use?")
            hdf5_filename, _ = process.extractOne(selected_file, glob.glob("*.hdf5"))

        else:
            hdf5_filename = input('What would you like to name the new hdf5 file?')
    # IF THERE ARE NO HDF5 FILES IN DIR, PROMPT USER TO MAKE ONE
    else:
        hdf5_filename = input('No existing hdf5 files detected. What would you like to name the new file?')
        hdf5_filename = hdf5_filename + '.hdf5'
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