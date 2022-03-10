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
from enum import Enum
from PIL import Image, ImageFilter



""" -- Constants -- """

# the various folders to look for images
NEW_FOLDER = "images_new_to_system/"
GENERATION_FOLDER = "tagged_images/"
ORIGINALS_FOLDER = "originals_in_system/"
DECODE_INPUT = "decoding/to_check_ownership/"
DECODE_OUTPUT = "decoding/owner_labeled/"

# the image extension to work wiht
EXTENSION = ".png"
# the maximum color value allowed
COLOR_MAX = 255
# the offset to the pixel color to mark a positive
COLOR_OFFSET = 10
# the percent of the image to pad on each side (assuming they won't cut off more than 1/6th)
PADDING_PCT = 1/6



""" -- Constants calculated from above -- """

# calculate the ranges we need to fit into
ALLOWED_COLOR_MIN = COLOR_OFFSET
ALLOWED_COLOR_MAX = COLOR_MAX - COLOR_OFFSET
ALLOWED_COLOR_RANGE = ALLOWED_COLOR_MAX - ALLOWED_COLOR_MIN
# and what percent of the image we can work in
REMAINING_PCT = 1 - PADDING_PCT - PADDING_PCT



""" -- ensuring the folders exist -- """

# all of our folders
for folder in [ NEW_FOLDER,
                GENERATION_FOLDER,
                ORIGINALS_FOLDER,
                DECODE_INPUT,
                DECODE_OUTPUT]:
    if not os.path.exists(folder):
        os.makedirs(folder)



""" -- Helper Functions -- """

def image_exists(folder: str, name: str) -> bool:
    """ check if an image exists in just our file system """
    if name[-len(EXTENSION):] != EXTENSION:
        name += EXTENSION
    return name in os.listdir(folder)


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


class Ternary:
    """ basic functionality to store a ternary value 3 trits long """
    def __init__(self, arg):
        if type(arg) is int:
            self.init_from_int(arg)
        elif type(arg) is tuple:
            self.init_from_trits(*arg)
        elif type(arg) is list:
            self.init_from_list(arg)
        elif type(arg) is str and len(arg) == 1:
            self.init_from_char(arg)
        else:
            raise TypeError("Unrecognized Ternary input, takes int, tuple, or list")
    
    def init_from_int(self, value):
        self.value = value
        self.low, self.mid, self.high = Ternary.int2tern(value)
        self.char = None
    
    def init_from_trits(self, high, mid, low):
        self.high = high
        self.mid = mid
        self.low = low
        self.value = Ternary.tern2int(high, mid, low)
        self.char = None
        
    def init_from_list(self, lst):
        self.high = lst[0]
        self.mid = lst[1]
        self.low = lst[2]
        self.value = Ternary.tern2int(lst[0], lst[1], lst[2])
        self.char = None
        
    def init_from_char(self, char):
        # just convert into a value
        if char.islower():
            self.char = char
            self.init_from_int(ord(char) - 96) # 97 = ord('a')
        elif char.isupper():
            self.char = char.lower()
            self.init_from_int(ord(char) - 64) # 65 = ord('A')
        elif char == " " or char == "_":
            self.char = "_"
            self.init_from_int(0) # [1-26]=>[a-z], [0]=>['_']
        else:
            raise ValueError("Unsupported character: " + char)
    
    def int2tern(val):
        low = val % 3
        val = val // 3
        mid = val % 3
        val = val // 3
        high = val % 3
        return low, mid, high
        
    def tern2int(high, mid, low):
        return 9*high + 3*mid + low
    
    def as_list(self):
        return [self.high, self.mid, self.low]
    
    def as_char(self):
        if self.value == 0:
            self.char = "_"
        else:
            self.char = chr(self.value + 96)
        return self.char
    
    def __str__(self):
        if self.char:
            return f"{self.char} (0t{self.high}{self.mid}{self.low})"
        else:
            return f"{self.value} (0t{self.high}{self.mid}{self.low})"


def mean_edge(horiz_edge: np.ndarray):
    """ given a horizontal edge of an image, finds the 6 averages of the sections """
    boxes = np.array_split(horiz_edge, 6, axis=1)
    return [(box.size, box.mean()) for box in boxes]


def get_box_dims(h: int, w: int, leng: int) -> tuple[int, int, int]:
    """
    given a box with a known width and height, determine the rows and columns
    needed to be able to fit <leng> squares in with them being as large as possible,
    as well as the extra padding squares left
    """
    
    # know that a max for the size is going to be fitting them all exactly in
    size = int(np.sqrt(h * w / leng)) + 1
    # the count of boxes we have
    count = 0
    
    while count <= leng and size > 0:
        size -= 1
        rows = h // size
        cols = w // size
        count = rows * cols
    
    return (rows, cols, count - leng)



def pop_extreme(img: np.ndarray, do_mimumum: bool = True) -> tuple[int, int, int]:
    """ find the location of the minimum/maximum value in the array and set it to 127 """
    # find the lowest
    ndx = img.argmin() if do_mimumum else img.argmax()
    # convert it to a multi-dimensional tuple
    r, c, _ = np.unravel_index(ndx, img.shape)
    # "pop" it out in all 3 color dimensions
    img[r, c, :] = img[r-1, c-1, :]
    
    return (r, c)


def decode_image(wild_name: str) -> str:
    """ decodes one image and figures out the tag for it """

    # do some prep work with the wild image
    # get the wild image
    wild_img = open_image(DECODE_INPUT, wild_name)
    wild_h, wild_w, _ = wild_img.shape
    
    # find the corners of the center block (lightest and darkest pixels)
    corners = np.array([pop_extreme(wild_img, mnmx) for _ in range(4) for mnmx in [True, False]])

    # find the actual corners
    # get the mean to find the "center"
    g_centers = corners.mean(axis=0)
    # split things into above or below the center
    g_greater_cent = corners > g_centers
    # get the median of our samples (to avoid outliers)
    g_top_ndx = int(np.median(corners[:, 0][~g_greater_cent[:, 0]]) + .5)# where the y's are not below center
    g_bottom_ndx = int(np.median(corners[:, 0][g_greater_cent[:, 0]]) - .5) # where the y's are below center
    g_right_ndx = int(np.median(corners[:, 1][g_greater_cent[:, 1]]) - .5) # where the x's are righter than center
    g_left_ndx = int(np.median(corners[:, 1][~g_greater_cent[:, 1]]) + .5) # where the x's are not righter than center


    # load in the original image
    # cut down to just the ending number
    clean_input_name = wild_name[wild_name.rfind("-") +1 : ] if ("-" in wild_name) else wild_name

    # open it
    lab_img = open_image(ORIGINALS_FOLDER, clean_input_name)
    lab_h, lab_w, _ = lab_img.shape
    
    # get the padding
    lab_pad_h = int(lab_h * REMAINING_PCT)
    lab_pad_w = int(lab_w * REMAINING_PCT)

    
    # map the two onto each other
    # get the wild size
    wild_pad_h = g_bottom_ndx - g_top_ndx
    wild_pad_w = g_right_ndx - g_left_ndx

    # going to scale the lab image to the wild's size
    lab_new_h = int(lab_h * (wild_pad_h / lab_pad_h))
    lab_new_w = int(lab_w * (wild_pad_w / lab_pad_w))
    lab_resized = np.array(Image.fromarray(lab_img).resize((lab_new_w, lab_new_h)))
    lab_new_pad_h = int(lab_new_h * REMAINING_PCT)
    lab_new_pad_w = int(lab_new_w * REMAINING_PCT)

    # figure out the width of the padding itself
    lab_top_pad = (lab_new_h - lab_new_pad_h) // 2
    lab_left_pad = (lab_new_w - lab_new_pad_w) // 2

    # how much we can cut off of the lab image to get it to the wild one
    top_cut = lab_top_pad - g_top_ndx
    left_cut = lab_left_pad - g_left_ndx

    # crop it
    lab_crop = lab_resized[top_cut : top_cut+wild_h, left_cut : left_cut+wild_w]

    # subtract the two to get the data
    found_mask = wild_img.astype(float) - lab_crop.astype(float)


    # extract the data from the edges
    # collect together all of the edges
    left_means = mean_edge(found_mask[g_top_ndx:g_bottom_ndx,  :g_left_ndx].transpose(1, 0, 2))
    right_means = mean_edge(found_mask[g_top_ndx:g_bottom_ndx, g_right_ndx: ].transpose(1, 0, 2))
    top_means = mean_edge(found_mask[ :g_top_ndx, g_left_ndx:g_right_ndx])
    bottom_means = mean_edge(found_mask[g_bottom_ndx: , g_left_ndx:g_right_ndx])

    edge_means = [left_means, right_means, top_means, bottom_means]

    # merge them together
    master_edge = []
    for i in range(6):
        total = sum(edge[i][0] * edge[i][1] for edge in edge_means)
        count = sum(edge[i][0] for edge in edge_means)
        master_edge.append(total / count)
        
    controls = master_edge[:3]
    def decode_val(item: float) -> int:
        return np.argmin([abs(item-cont) for cont in controls])

    # decode the length of the string now
    wild_name_len = Ternary([decode_val(test) for test in master_edge[3:]]).value + 1
    # plus one since we go from 1-27 rather than 0-26


    # extract the actual data
    # find rows and columns for the the lab image so the integer math works out the same
    wild_rows, wild_cols, _ = get_box_dims(lab_pad_h, lab_pad_w, wild_name_len * 3)

    # pull out the actual data cells
    center_data = found_mask[g_top_ndx:g_bottom_ndx, g_left_ndx:g_right_ndx].mean(axis=2) # mean over the color dimension

    # split it into a grid
    found_cells = [np.array_split(row, wild_cols, axis=1) for row in np.array_split(center_data, wild_rows, axis=0)]
    
    # average them all and flatten them into a single list
    found_data = [cell.mean() for row in found_cells for cell in row]

    # find the closest tag for each
    wild_tag = [decode_val(sample) for sample in found_data][:wild_name_len*3]
    
    # split it into 3s and convert them to characters
    name_chars = [Ternary(wild_tag[3*i:3*i +3]).as_char() for i in range(wild_name_len)]
    
    # convert back into a word
    found_name = "".join(name_chars)

    return found_name



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
    
    # prep the name
    # make sure the name is valid
    assert len(person_name) <= 27, "Name must be 27 characters or shorter!"
    assert len(person_name) > 0, "Must actually provide a name!"
    name = person_name.lower().replace(" ", "_")
    assert all(let == "_" or let.isalpha() for let in name), "Name has something other than a letter, space, or underscore"

    # we'll need 3 spots for each letter
    num_spots = len(name) * 3


    # prep the image
    # open the image
    assert image_exists(ORIGINALS_FOLDER, img_name), "Image must exist in the file system"
    img = open_image(ORIGINALS_FOLDER, img_name)
    h, w, _ = img.shape
    
    # cut the size down by the designated padding
    pad_h = int(h * REMAINING_PCT)
    pad_w = int(w * REMAINING_PCT)

    
    # make the center name data
    # get the box sizes and padding we need
    rows, cols, box_padding = get_box_dims(pad_h, pad_w, num_spots)

    # convert the name into a long string of trits
    name_trits = [trit for letter in name for trit in Ternary(letter).as_list()]

    # append our padding to the end
    ternary_data = name_trits + [0] * box_padding

    # turn this data into a matrix of color offsets
    # make it a numpy array
    offsets = np.array(ternary_data)
    # reshape it into our dimensions
    offsets = offsets.reshape(rows, cols)
    # some math to make (0, 1, 2) into (1, 2, 0)
    offsets = (offsets + 1) % 3 
    # multiply by the offset
    offsets *= COLOR_OFFSET
    # repeat it so we have 3 color channels
    offsets = np.repeat(offsets[:, :, np.newaxis], 3, axis=2)

    # convert the array into an image temporarily so we can scale it up easily
    as_img = Image.fromarray(offsets.astype(np.uint8))
    as_img = as_img.resize((pad_w, pad_h), resample=Image.NEAREST)
    offsets = np.array(as_img)


    # make the edges/padding of control values
    # figure out the pixel size of the padding itself
    top_pad = (h - pad_h) // 2
    left_pad = (w - pad_w) // 2

    # make the string of control values, reference values and the length of the name
    # subtract one from the name length to do 1-27 rather than 0-26
    control_data = [0, 1, 2] + Ternary(len(name) - 1).as_list()

    # make our lists of arrays to store these values
    # have it as long as the padding, then split into 6 sections
    across = np.array_split(np.zeros(pad_w), 6)
    # and as tall as padding, split into 6 as well
    up = np.array_split(np.zeros(pad_h), 6)

    # fill each block with the right data
    for i in range(6):
        across[i] += control_data[i]
        up[i] += control_data[i]

    # concatenate our arrays back together
    control_across = np.concatenate(across)
    control_up = np.concatenate(up)
    
    # do math to scale them into the right ranges
    control_across = ((control_across + 1) % 3 )
    control_across *= COLOR_OFFSET
    control_up = ((control_up + 1) % 3 )
    control_up *= COLOR_OFFSET

    # duplicate them into the dimensions they need
    # repeat the horizontal line in both the vertical direction and the color direction
    control_top = np.repeat(np.repeat(control_across[np.newaxis, :], repeats=(top_pad), axis=0)[:, :, np.newaxis], repeats=3, axis=2)
    control_bottom = np.repeat(np.repeat(control_across[np.newaxis, :], repeats=(h - top_pad - pad_h), axis=0)[:, :, np.newaxis], repeats=3, axis=2)
    # repeat this line in both the horizontal direction and the color direction
    control_left = np.repeat(np.repeat(control_up[:, np.newaxis], repeats=(left_pad), axis=1)[:, :, np.newaxis], repeats=3, axis=2)
    control_right = np.repeat(np.repeat(control_up[:, np.newaxis], repeats=(w - left_pad - pad_w), axis=1)[:, :, np.newaxis], repeats=3, axis=2)


    # combine everything back together into a single mask
    # the base image should be 0 change, but right now we're shifted up by COLOR_OFFSET
    mask = np.zeros(img.shape, dtype=np.uint8) + COLOR_OFFSET

    # add in our border around it
    mask[top_pad:top_pad+pad_h, 0:left_pad, :] = control_left
    mask[top_pad:top_pad+pad_h, left_pad+pad_w:, :] = control_right
    mask[0:top_pad, left_pad:left_pad+pad_w, :] = control_top
    mask[top_pad+pad_h:, left_pad:left_pad+pad_w, :] = control_bottom

    # put our offsets into the middle
    mask[top_pad:top_pad+pad_h, left_pad:left_pad+pad_w, :] = offsets

    # blur our mask
    radius = int(min([pad_h/rows, pad_w/cols, pad_h/6, pad_w/6]) / 3)
    mask = np.array(Image.fromarray(mask).filter(ImageFilter.BoxBlur(radius=radius))) # (box) size is 57


    # create the final image
    # merge the original image and the mask, making sure to watch for the numbers signs
    out_img = (img.copy().astype(int) + mask - COLOR_OFFSET).astype(np.uint8)

    # mark the corners with a single pure white pixel on the horizontal outside of the bounds
    for i in [-1, 1]:
        for j in [-1, 1]:
            out_img[i * top_pad, j*(left_pad-1)] = 255
    # mark the corners with a single pure black pixel on the vertical outside of the bounds
    for i in [-1, 1]:
        for j in [-1, 1]:
            out_img[i * (top_pad-1), j*left_pad] = 0
    

    # give back the results
    # convert back to an image and save it
    tagged_image_name = name + "-" + img_name
    Image.fromarray(out_img).save(GENERATION_FOLDER + tagged_image_name)

    return tagged_image_name



def decode_images(print_mode: bool) -> None:
    """ goes through our decoder's input folder and figures out who took them all """
    
    decode_inputs = os.listdir(DECODE_INPUT)

    if print_mode and len(decode_inputs) == 0:
        print("No images to decode! Put them in the folder", DECODE_INPUT)
        return

    for input_name in decode_inputs:
        if print_mode:
            print("Decoding image", input_name)

        clean_input_name = input_name[input_name.rfind("_") +1 : ] if ("_" in input_name) else input_name

        img_tag = decode_image(input_name)

        if print_mode:
            print("Owner found to be", img_tag)

        # label the image
        os.rename(DECODE_INPUT + input_name, DECODE_OUTPUT + img_tag + "-" + input_name)


def remove_image(img_num: str) -> None:
    """ removes the given image from our database, originals folder, and tagged folder """

    # first remove our tagged images
    all_tagged = os.listdir(GENERATION_FOLDER)
    def get_num(img_name):
        return img_name[img_name.rfind("-") +1 : -len(EXTENSION)]
    for image in all_tagged:
        if get_num(image) == img_num:
            os.remove(GENERATION_FOLDER + image)

    # remove from our originals folder
    os.remove(ORIGINALS_FOLDER + img_num + EXTENSION)



""" -- Text-Based Menu Functions -- """

def menu_standardize_images():
    """move images from new folder to generated folder, giving them numerical names"""
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
        if img_name[-4:] != EXTENSION:
            img_name += EXTENSION
        img_valid = image_exists(ORIGINALS_FOLDER, img_name)
        if not img_valid:
            print("Image must exist in the file system")

    keep_whiling = True
    print("What names should we log it under?")
    person_name = input()

    while keep_whiling:
        try:
            new_img_name = tag_image(img_name, person_name)
        except AssertionError as e:
            print(e)
            return
        except:
            print("An unknown error has occurred!")
            return

        print("Generated a new version of image", img_name, "for", person_name + ". It's named", new_img_name)

        print("Next name? (leave it blank to end)")
        person_name = input()
        keep_whiling = (person_name != "")

    # done


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


def check_menu() -> None:
    """ Prints the menu for the user and gets them to choose an option """
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
        check_menu()