"""
Author: Jake Swartwout
Date: 12/25/2021
Brief: Changes the color values of images to track who it is given out to
Description: By altering the brightness of 6 sections of an image, (3 control
areas, and 3 values, with either no alteration, increasing, or decreasing)
we can effectively encode 27 different unique values into an image which is
difficult to notice by a human but possible to check digitally. This allows
one to log who an image was originally given to, in order to track where it
was stolen from, even if the metadata is wiped, or the image is otherwise
changed, such as resizing, screenshotting, or compression.
"""


""" -- Imports -- """
import sys
import os
import numpy as np
import pandas as pd
from enum import Enum
from PIL import Image, ImageFilter



""" -- Constants -- """

# the various folders to look for images
NEW_FOLDER = "images_new_to_system/"
GENERATION_FOLDER = "tagged_images/"
ORIGINALS_FOLDER = "originals_in_system/"
DECODE_INPUT = "decoding/to_check_ownership/"
DECODE_OUTPUT = "decoding/owner_labeled/"
# where our mappings are stored
MAPPINGS_FILE = "mapping.csv"
# the image extension to work wiht
EXTENSION = ".png"
# the maximum color value allowed
COLOR_MAX = 255
# the offset to the pixel color to mark a positive
COLOR_OFFSET = 10
# how far from the desired value the pixel can be colored
COLOR_RANGE = 5
# the column names
COL_IMG_NUM = "ImgNum"
COL_TAG_NUM = "TagNum"
COL_OWNER = "Owner"



""" -- Constants calculated from above -- """

# calculate the ranges we need to fit into
ALLOWED_COLOR_MIN = COLOR_OFFSET + COLOR_RANGE
ALLOWED_COLOR_MAX = COLOR_MAX - ALLOWED_COLOR_MIN
ALLOWED_COLOR_RANGE = ALLOWED_COLOR_MAX - ALLOWED_COLOR_MIN

# a global variable to only load the csv once
# the mappings dataframe to store the tags to people for each image
global_mappings = None



""" -- Helper Functions -- """

def get_mappings():
    """ singleton pattern to get the mappings csv or load it in if needed """
    global global_mappings
    if global_mappings is None:
        global_mappings = pd.read_csv(MAPPINGS_FILE)
    return global_mappings


def mappings_append(owner: str, img_num: str, tag_num: str) -> None:
    """ edits the global mappings dataframe to append the given data """
    global global_mappings
    data_vals = {COL_OWNER: owner, COL_IMG_NUM: int(img_num), COL_TAG_NUM: int(tag_num)}
    global_mappings = get_mappings().append(data_vals, ignore_index=True)
    # save the new mappings
    global_mappings.to_csv(MAPPINGS_FILE, index=False)


def mappings_remove(img_num: str) -> None:
    """ removes the given image from the mappings dataframe """
    global global_mappings
    mappings = get_mappings()
    not_matching_img = mappings[COL_IMG_NUM] != int(img_num)
    global_mappings = mappings[not_matching_img]
    global_mappings.to_csv(MAPPINGS_FILE, index=False)


def image_exists(folder: str, name: str) -> bool:
    """ check if an image exists in just our file system """
    if name[:-len(EXTENSION)] != EXTENSION:
        name += EXTENSION
    return name in os.listdir(folder)


def image_exists_both(folder: str, name: str) -> bool:
    """ check if an image exists both in our folder and in our database """
    if name[:-len(EXTENSION)] == EXTENSION:
        name_png = name
        name_num = name[:[-len(EXTENSION)]]
    else:
        name_png = name + EXTENSION
        name_num = name
    if name_png not in os.listdir(folder):
        return False
    if name_num not in get_mappings()[COL_IMG_NUM].apply(str):
        print("mappings")
        return False
    return True


def open_image(folder: str, name: str) -> np.ndarray:
    """ opens the image at the given location and returns a copy of its pixels as a numpy array """
    return np.asarray(Image.open(folder + name).convert("RGB")).copy()


def standardize_image(orig_img: np.ndarray) -> np.ndarray:
    """ Takes an image and ensures it's within the correct bounds for standardization """

    # get where it's currently at
    img_min = orig_img.min()
    img_max = orig_img.max()
    img_range = img_max - img_min

    # make a copy to modify
    img = orig_img.copy()
    
    # if the range larger than we allow, need to squish it
    if img_range > ALLOWED_COLOR_RANGE:
        img -= img_min # shift down to 0
        img = img * (ALLOWED_COLOR_RANGE / img_range) # scale it down to the right width
        img += ALLOWED_COLOR_MIN # shift it up to the right spot
        img = img.astype(np.uint8) # get rid of the decimals
    else:
        # small enough range, just make sure it's in bounds
        if img_min < ALLOWED_COLOR_MIN:
            img += (ALLOWED_COLOR_MIN - img_min)
        elif img_max > ALLOWED_COLOR_MAX:
            img -= (ALLOWED_COLOR_MAX - img_max)
        else:
            # should already be perfectly in bounds
            pass

    return img


def find_next_open_img() -> int:
    """ Goes through the originals folder and finds the next available spot """
    old_generated = os.listdir(ORIGINALS_FOLDER)

    next_open = None
    if len(old_generated) == 0:
        next_open = 1
    else:
        next_open = 1 + max(int(name[:-4]) for name in old_generated)

    return next_open


def get_next_tag(img_num: str, person_name: str) -> int:
    """
    searches the mappings file for the next available tag for a certain image,
    then claims that for this person
    """
    # load in the current values
    mappings = get_mappings()
    # find the next tag for that image
    matching_image = mappings[COL_IMG_NUM] == int(img_num)
    if not any(matching_image):
        next_tag = 1 # don't do 0, idk why it just seems boring
    else:
        next_tag = 1 + max(mappings[matching_image][COL_TAG_NUM])
    # add the ID paired with their name
    mappings_append(person_name, img_num, next_tag)
    return next_tag


class Ternary:
    """ basic functionality to store a ternary value 3 trits long """
    def __init__(self, arg):
        if type(arg) is int:
            self.init_from_int(arg)
        elif type(arg) is tuple:
            self.init_from_trits(*arg)
        elif type(arg) is list:
            self.init_from_list(arg)
        else:
            raise TypeError("Unrecognized Ternary input, takes int, tuple, or list")
    
    def init_from_int(self, value):
        self.value = value
        self.low, self.mid, self.high = Ternary.int2tern(value)
    
    def init_from_trits(self, high, mid, low):
        self.high = high
        self.mid = mid
        self.low = low
        self.value = Ternary.tern2int(high, mid, low)
        
    def init_from_list(self, lst):
        self.high = lst[0]
        self.mid = lst[1]
        self.low = lst[2]
        self.value = Ternary.tern2int(lst[0], lst[1], lst[2])
    
    def int2tern(val):
        low = val % 3
        val = val // 3
        mid = val % 3
        val = val // 3
        high = val % 3
        return low, mid, high
        
    def tern2int(high, mid, low):
        return 9*high + 3*mid + low
    
    def __str__(self):
        return f"{self.value} (0t{self.high}{self.mid}{self.low})"
    

def gen_rand(high: int, wide: int, trit: int) -> np.ndarray:
    """
    generates a random grid with high*rows and wide*columns with a range of 2 x COLOR_RANGE.
    if the trit is 0, they are left between (-range, range)
    if the trit is 1, they are offset to between (offset - range, offset + range)
    if the trit is 2, they are offset to between (-offset - range, -offset + range)
    """
    # generate the base random grid from (-range, range)
    rand = np.random.rand(high, wide, 3) * (COLOR_RANGE * 2) - COLOR_RANGE
    # offset it as necessary
    if trit == 1:
        rand += COLOR_OFFSET
    elif trit == 2:
        rand -= COLOR_OFFSET
#     elif trit == 0
#         rand += 0
    elif trit != 0:
        raise ValueError("Trit must be 0, 1, or 2")
    return rand


def get_block_sizes(a: int, b: int, is_vertical: bool = None) -> tuple[bool, tuple[int, int], tuple[int, int]]:
    """
    given the two dimensions of an image, return the sizes of each block it should be split into
    if a value is passed for is_vertical, force the image to be that direction, otherwise calculate it
    """
    wide = max(a, b)
    high = min(a, b)
    vertical = (wide == a) if (is_vertical is None) else is_vertical
    h1 = high // 2
    h2 = high - h1
    w1_2 = wide // 3
    w3 = wide - w1_2 - w1_2
    return (vertical, (h1, h2), (w1_2, w3))


def find_section_average(img: Image, orig_vertical: bool = None) -> tuple[np.ndarray, bool]:
    """
    given an image, find the averages for each section and return it in a (2, 3) numpy array
    if a value is passed for orig_vertical, force the image to be that direction, otherwise calculate it
    """
    # get the block sizes
    h, w, _ = img.shape
    vertical, (h_1, h_2), (w_1_2, w_3) = get_block_sizes(h, w, orig_vertical)

    # flip the image if its vertical
    if vertical:
        img = img.transpose(1, 0, 2)
    
    # to hold our values as we find them
    averages = np.zeros((2, 3))
    
    # the control blocks
    # zero / none
    averages[0, 0] = img[:h_1, :w_1_2, :].mean()
    # one / positive
    averages[0, 1] = img[:h_1, w_1_2:2*w_1_2, :].mean()
    # two / negative
    averages[0, 2] = img[:h_1, 2*w_1_2:, :].mean()
    
    # the encoded blocks
    # low
    averages[1, 0] = img[h_1:, :w_1_2, :].mean()
    # mid
    averages[1, 1] = img[h_1:, w_1_2:2*w_1_2, :].mean()
    # high
    averages[1, 2] = img[h_1:, 2*w_1_2:, :].mean()
    
    # convert it to integers
    return (averages.astype(int), vertical)


def decode_image(input_name: str) -> Ternary:
    """ decodes one image and figures out the tag for it """
    
    # cut down to the ending number
    clean_input_name = input_name[input_name.rfind("_") +1 : ] if ("_" in input_name) else input_name
    # cut off the extension
    input_num = clean_input_name[:clean_input_name.rfind(".")]

    # find the average color values for the original image
    lab_img = open_image(ORIGINALS_FOLDER, clean_input_name)
    lab_avgs, orig_vertical = find_section_average(lab_img)

    # find the average color values for the wild image
    wild_img = open_image(DECODE_INPUT, input_num + EXTENSION)
    wild_avgs, _ = find_section_average(wild_img, orig_vertical)

    # compare the two
    delta_avgs = wild_avgs - lab_avgs
    delta_avgs -= delta_avgs[0, 0]

    # find the best match for each item
    wild_tag = [abs(delta_avgs[0] - sample).argmin() for sample in delta_avgs[1]]
    
    # convert it to a ternary output in the right order
    return Ternary(tuple(reversed(wild_tag)))



""" -- Functions to Export -- """

def standardize_images(print_mode: bool) -> None:
    """
    goes through our folder of new images and squishes the colors into the
    correct range for easier manipulating later
    """
    imgs_to_convert = os.listdir(NEW_FOLDER)

    if print_mode and len(imgs_to_convert) == 0:
        print("No new images to convert, make sure to put them in", NEW_FOLDER)

    for img_name in imgs_to_convert:
        if print_mode:
            print("Standardizing", img_name)

        # open it (in RBG mode)
        img = open_image(NEW_FOLDER, img_name)
        
        # standardize it to between our bounds
        bounded_img = standardize_image(img)

        # find the next number we can name it
        next_num = find_next_open_img()

        # save to our folder as our extension
        Image.fromarray(bounded_img).save(ORIGINALS_FOLDER + str(next_num) + EXTENSION)

        # delete the image from the new-images folder
        os.remove(NEW_FOLDER + img_name)


def tag_image(img_name: str, person_name: str) -> str:
    """ generates a newly tagged version of that image for the requested person """
    assert image_exists(ORIGINALS_FOLDER, img_name), "Image must exist in the file system"

    #get the next tag
    next_tag = get_next_tag(img_name, person_name)
    tag = Ternary(next_tag)
    
    # open the image
    img = open_image(ORIGINALS_FOLDER, img_name + EXTENSION)
    h, w, _ = img.shape
    
    # get the sizes of the blocks
    vertical, (h_1, h_2), (w_1_2, w_3) = get_block_sizes(h, w)
    
    # create the randomized blocks
    rand = np.zeros(((h_1+h_2, w_1_2+w_1_2+w_3, 3)))
    # the control blocks
    # zero / none
    rand[:h_1, :w_1_2, :] = gen_rand(h_1, w_1_2, 0)
    # one / positive
    rand[:h_1, w_1_2:2*w_1_2, :] = gen_rand(h_1, w_1_2, 1)
    # two / negative
    rand[:h_1, 2*w_1_2:, :] = gen_rand(h_1, w_3, 2)
    # the encoded blocks
    # low
    rand[h_1:, :w_1_2, :] = gen_rand(h_2, w_1_2, tag.low)
    # mid
    rand[h_1:, w_1_2:2*w_1_2, :] = gen_rand(h_2, w_1_2, tag.mid)
    # high
    rand[h_1:, 2*w_1_2:, :] = gen_rand(h_2, w_3, tag.high)
    # flip it if necessary
    if vertical:
        rand = rand.transpose(1, 0, 2)
    
    # blur them together
    # convert the format to use PIL's BoxBlur
    blur = Image.fromarray((rand + ALLOWED_COLOR_MIN).astype(np.uint8))
    rad = min(w,h) / 6
    blur = blur.filter( ImageFilter.BoxBlur(radius=rad) )
    
    # add the randomness to the image
    done = img + np.asarray(blur) - ALLOWED_COLOR_MIN
    
    # save the image
    tagged_image_name = person_name + "_" + img_name + EXTENSION
    Image.fromarray(done).save(GENERATION_FOLDER + tagged_image_name)
    
    # tell them where it is
    return tagged_image_name


def decode_images(print_mode: bool) -> None:
    """ goes through our decoder's input folder and figures out who took them all """
    mappings = get_mappings()

    decode_inputs = os.listdir(DECODE_INPUT)

    if print_mode and len(decode_inputs) == 0:
        print("No images to decode! Put them in the folder", DECODE_INPUT)
        return

    for input_name in decode_inputs:
        if print_mode:
            print("Decoding image", input_name)

        clean_input_name = input_name[input_name.rfind("_") +1 : ] if ("_" in input_name) else input_name

        img_tag = decode_image(input_name)

        where_img = ( mappings["ImgNum"] == int(clean_input_name[:-len(EXTENSION)]) )
        where_tag = ( mappings["TagNum"] == img_tag.value )
        owners = mappings[where_img & where_tag]["Owner"]
        assert len(owners) == 1, "Found the wrong number of image owners! Found: " + str(len(owners)) + " for img " + input_name + " and tag " + str(img_tag)
        found_owner = owners.values[0]

        if print_mode:
            print("Owner found to be", found_owner)

        # label the image
        user_name = input_name[:-(1 + len(clean_input_name))] # get the first, user-given part of the name
        new_name = user_name + "(Owner-" + found_owner + ")_" + clean_input_name
        os.rename(DECODE_INPUT + input_name, DECODE_OUTPUT + new_name)


def remove_image(img_num: str) -> None:
    """ removes the given image from our database, originals folder, and tagged folder """

    # first remove our tagged images
    all_tagged = os.listdir(GENERATION_FOLDER)
    def get_num(img_name):
        return img_name[img_name.rfind("_") +1 : -len(EXTENSION)]
    for image in all_tagged:
        if get_num(image) == img_num:
            os.remove(GENERATION_FOLDER + image)

    # remove from the database
    mappings_remove(img_num)

    # remove from our originals folder
    os.remove(ORIGINALS_FOLDER + img_num + EXTENSION)



""" -- Menu Functions -- """

def menu_standardize_images():
    """move images from new to protected, giving them numerical names"""
    print("Standardizing images")
    standardize_images(True)


def menu_generate_new_id():
    """asks which image and the name, then associates the two together """
    print("Running the image tagging wizard...")

    if len(os.listdir(ORIGINALS_FOLDER)) == 0:
        print("We don't even have any images in the system yet!")
        return
    
    # get the image to work with and the name to save it under
    print("Which image should we encode?")
    img_valid = False
    while not img_valid:
        img_name = input()
        if img_name[-4:] == EXTENSION:
            img_name = img_name[:-len(EXTENSION)]
        img_valid = image_exists(ORIGINALS_FOLDER, img_name)
        if not img_valid:
            print("Image must exist in the file system")

    print("What name should we log it under?")
    person_name = input()

    new_img_name = tag_image(img_name, person_name)

    print("Generated a new version of image", img_name, "for", person_name + ". It's named", new_img_name)


def menu_check_id():
    """goes through the decoder inputs and labels them with the owners"""
    print("Checking the ID for all given images...")
    decode_images(True)


def menu_delete_img():
    """asks which image to delete from the system and does so"""
    print("Running the image deletion wizard...")

    if len(os.listdir(ORIGINALS_FOLDER)) == 0:
        print("We don't even have any images in the system yet!")
        return
    
    # get the image to work with and the name to save it under
    print("Which image do you want to delete?")
    img_valid = False
    while not img_valid:
        img_name = input()
        if img_name[-4:] == EXTENSION:
            img_name = img_name[:-len(EXTENSION)]
        img_valid = image_exists(ORIGINALS_FOLDER, img_name)
        if not img_valid:
            print("Image must exist in the file system")

    remove_image(img_name)


def menu_exit():
    """Exits the program"""
    print("Exiting...")
    sys.exit(0)
    return



""" -- Handling the menu -- """

class Menu(Enum):
    """Options the user can choose in the menu"""
    StandardizeImages = ("Standardize the images", menu_standardize_images, "Standardize Images")
    GenerateNewID = ("Generate a new tag for an image", menu_generate_new_id, "Generate a New Tag")
    CheckID = ("Check the IDs for a set of images", menu_check_id, "Check Tags")
    DeleteImage = ("Completely delete an image from the system", menu_delete_img, "Delete an Image")
    Exit = ("Exit the program", menu_exit, "Exit")

# a dict mapping numbers to menu items
menu = {str(i+1): item for i, item in enumerate(Menu)}


def check_menu():
    """Prints the menu for the user and gets them to choose an option"""
    # print the menu
    print("-"*15)
    for option in menu:
        print(f"{option}: {menu[option].value[0]}")
    # get input
    answer = input()
    # repeatedly ask until they pick a valid choice
    while answer not in menu:
        print("That's not a valid menu option")
        answer = input()
    # call the function
    menu[answer].value[1]()



""" -- Main function -- """

if __name__ == "__main__":
    while True:
        user_choice = check_menu()