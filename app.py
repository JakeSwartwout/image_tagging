"""
Author: Jake Swartwout
Date: 12/27/2021
Brief: A simple web frontend to interact with the Image Tagger
Description: Same as brief
"""



""" -- Imports -- """

import os
from ImageTagger import menu, standardize_images, tag_image, decode_images, remove_image
from ImageTagger import NEW_FOLDER, GENERATION_FOLDER, ORIGINALS_FOLDER, DECODE_INPUT, DECODE_OUTPUT, MAPPINGS_FILE, EXTENSION
from flask import Flask, url_for, render_template, request, redirect



""" -- CONSTANTS -- """

HTML_EXTENSION = ".html"
EXT = lambda name : name + HTML_EXTENSION


""" -- Helper functions -- """

def read(filename):
    """ opens the html file at the given location and returns the contents """
    with open(filename, 'r', encoding="utf-8") as page:
        return page.read()



""" -- App code -- """
app = Flask(__name__)

@app.route("/")
def root():
    return redirect(url_for('home'))

@app.route("/home")
def home():
    return render_template(EXT('home'), menu=menu)



""" -- Menu Functions -- """

@app.route("/menu/standardize")
def menu_standardize_images():
    """move images from new to protected, giving them numerical names"""
    standardize_images(False)
    return "standardize"

@app.route("/menu/new_tag")
def menu_generate_new_id():
    """asks which image and the name, then associates the two together """
    # return render_template(EXT(' '), os.listdir(ORIGINALS_FOLDER))
    return "new_tag"

@app.route("/menu/check_id")
def menu_check_id():
    """goes through the decoder inputs and labels them with the owners"""
    # print("Checking the ID for all given images...")
    # decode_images(True)
    return "check_id"

@app.route("/menu/delete_img")
def menu_delete_img():
    """asks which image to delete from the system and does so"""
    return f"delete_img"
    # remove_image(img_name)

@app.route("/menu/close")
def menu_exit():
    """Exits the program"""
    return "exit"



if __name__ == "__main__":
    app.run()