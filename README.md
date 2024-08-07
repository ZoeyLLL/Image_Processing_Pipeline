#Image Processing Pipeline

Use the environment specified in pipfile and pipfile.lock to run this project.
Modules of numpy, skimage, matplotlib need to be installed using pip3 install commands. 

To start the project, in terminal, run
 python3 main.py

In terminal it will ask Automatic or Manual? (a/m)

### 1.1 
If you want to get images from 1.1 automatic pipeline, type a and hit return. 

There will be a plot showing up having the comparison of gray world, white world and preset white balancing images in each step of the ISP pipeline.

After you close the plot window, the project will start to download the 6 jpg and png files using gray-world white-balanced images (baby_gray.png, baby_gray.jpeg), white-world white-balanced images (baby_white.png, baby_white.jpeg) and preset white-balanced images (baby_preset.png, baby_preset.jpeg)


### 1.2
If you want to get images from manually choosing the white spot of the image, type m and hit run.

The plot will show up with instruction 'Click on a point that should be white'.

Double click to select the point where you consider it as the white. 

There will be a plot showing up having the image of each stage of ISP pipeline that set the point you selected as white for white-balancing. 

After you close the plot window, the project will download the png image 'baby_white_manual.png'. 

If you want to try different batches by selecting different spot as white, you can change the file name and repeat the above steps. 

### 1.3 
Commands are here for you to type in the terminal: 

To get the ppm file: 
dcraw -v -a -c -q 0 -o 1 -b 2 ./data/baby.nef > ./data/baby.ppm

To get the png file by netbpm:
First install netbpm:
brew install Netpbm

Then write this command in terminal:
pnmtopng ./data/baby.ppm ./data/baby.png


