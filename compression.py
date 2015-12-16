import random

from PIL import Image


def openImage(path):
    return Image.open(path)


def getRandomSquare(img):
    """Get random square 8x8 from the picture
       and quantify their colours to <0;1>
    """
    imgBW = img.convert('L')
    x = random.randint(0, (imgBW.size[0] - 8))
    y = random.randint(0, (imgBW.size[1] - 8))

    rows = []
    for i in range(8):
        for j in range(8):
            rows.append(imgBW.getpixel((x + j, y + i)))

    for i in range(len(rows)):
        rows[i] /= 255.0
    return rows


def getFixedSquare(img, x, y):
    """Get 8x8 squares from the fixed position of  the picture
    """
    imgBW = img.convert('L')
    rows = []
    for i in range(8):
        for j in range(8):
            rows.append(imgBW.getpixel((x + j, y + i)))

    for i in range(len(rows)):
        rows[i] /= 255.0
    return rows


def quantify(layer, bits):
    """Quantify real number output of layer to the no. of bits
    """
    variants = pow(2, bits)
    step = 1.0 / variants
    quant = []
    for i in layer:
        for j in range(variants):
            if j == variants - 1:
                # max possible value no matter which interval
                ret = bin(0 + j)[2:].zfill(bits)
                quant.append(ret)
                break
            elif 0.0 + j * step <= i < 0.0 + (j + 1) * step:
                ret = bin(0 + j)[2:].zfill(bits)
                quant.append(ret)
                break

    return quant


def dequantify(quant):
    """Dequantifies list of bit values
    """
    if not len(quant):
        print 'Empty list'
        exit(-1)

    step = 1.0 / pow(2, len(quant[0]))
    x = int(quant[0], 2)
    dequant = []

    for i in quant:
        val = int('0b' + i, 2)
        ret = step / 2.0 + val * step
        dequant.append(ret)

    return dequant


def putSquare(img, x, y, real_values, filename):
    """Generate 8x8 square into given pos image from list of 64 realValues containing numbers 0.0-1.0
       Saves under fileName.
    """
    quant = quantify(real_values, 8)
    idx = 0
    for a in range(8):
        for b in range(8):
            img.putpixel((x + b, y + a), int('0b' + quant[idx], 2))
            idx += 1
    img.save(filename, 'BMP')


def getSequenceSquares(img):
    """Get all consecutive squares from the picture
    """
    x = img.size[0]
    y = img.size[1]
    squares = []
    for i in xrange(0, x, 8):
        for j in xrange(0, y, 8):
            squares.append(getFixedSquare(img, i, j))
    return squares


def printPicture(img, squares, name):
    """Print squares sequence into img
    """
    img = img.convert('L')
    idx = 0
    x = img.size[0]
    y = img.size[1]
    for i in xrange(0, x, 8):
        for j in xrange(0, y, 8):
            putSquare(img, i, j, squares[idx], name)
            idx += 1