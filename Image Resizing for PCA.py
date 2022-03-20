from PIL import Image
import os, sys

dir="C:/Users/Kunj/Downloads/raw-img/"
path = "squirrel/" # Folder path to be resized
dirs = os.listdir(dir+path)

def resize():
    count = 0 
    for item in dirs:
        if os.path.isfile(dir+path+item):
#             if imagePath == directory + '.DS_Store':
#                 continue
        
            im = Image.open(dir+path+item)
            
            
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            f, e = os.path.splitext(path+item)
            imResize = im.resize((25,25), Image.ANTIALIAS)
            imResize.save( dir+'resized_animals/' + f + 'resized.jpg', 'JPEG', quality=100)
            count+=1

resize()