#!/usr/bin/env python
import sys
import random
from PIL import Image


def openImage(path):
    try:
        img = Image.open(path)
    except:
        print "Unable to load image";
        exit(-1)
    return img



def getRandomSquare(img):
    """Get random square 8x8 from the picture
    and quantify their colours to <0;1>"""
    imgBW = img.convert('L')
    x = random.randint(0, (imgBW.size[0]-8))
    y = random.randint(0, (imgBW.size[1]-8))

    rows = []
    for i in range(8):
        for j in range(8):
            rows.append(imgBW.getpixel((x+i, y+j)))

    for i in range(len(rows)):
        rows[i] = rows[i]/255.0
    return rows

def getFixedSquare(img, x, y):
    """Get 8x8 squares from the fixed position of  the picture"""
    imgBW = img.convert('L')
    rows = []
    for i in range(8):
        for j in range(8):
            rows.append(imgBW.getpixel((x+i, y+j)))

    for i in range(len(rows)):
        rows[i] = rows[i]/255.0
    return rows

def quantify(layer, bits):
    """Quantify real number output of layer to the no. of bits"""
    step = 1.0/(pow(2, bits))

    quant = []
    for i in layer:
        for j in range(pow(2, bits)):
            if(0.0+j*step <= i < 0.0 +(j+1)*step):
                ret = bin(0+j)[2:].zfill(bits)
                quant.append(ret)

    return quant

def dequantify(quant):
    """Dequantifies list of bit values"""
    if not len(quant):
       print 'Empty list'
       exit(-1)

    step = 1.0/pow(2, len(quant[0]))
    x = int(quant[0], 2)
    dequant = []

    for i in quant:
        val = int("0b"+i, 2)
        ret = step/2.0 + val*step
        dequant.append(ret)


    return dequant

def putSquare(img, x, y, realValues):
    """
    Generate 8x8 square into given pos image from list of 64 realValues containing numbers 0.0-1.0
    """
    img = img.convert('L')
    quant = quantify(realValues, 8)
    for i in quant:
        for a in range(8):
            for b in range(8):
                img.putpixel((x+a, y+b), int("0b"+i, 2))

    img.save("newimg", "BMP")


putSquare(openImage("1.bmp"), 0, 0, getFixedSquare(openImage(sys.argv[1]), 0, 0))
