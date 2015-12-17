import random

from PIL import Image


def open_image(path):
    return Image.open(path)


def new_image(size):
    return Image.new('F', size)


def get_random_square(img):
    """Get random square 8x8 from the picture
       and quantify their colours to <0;1>
    """
    img_grayscale = img.convert('L')
    x = random.randint(0, (img_grayscale.size[0] - 8))
    y = random.randint(0, (img_grayscale.size[1] - 8))

    rows = []
    for i in range(8):
        for j in range(8):
            rows.append(img_grayscale.getpixel((x + j, y + i)))

    for i in range(len(rows)):
        rows[i] /= 255.0
    return rows


def get_fixed_square(img, x, y):
    """Get 8x8 squares from the fixed position of  the picture
    """
    img_grayscale = img.convert('L')
    rows = []
    for i in range(8):
        for j in range(8):
            rows.append(img_grayscale.getpixel((x + j, y + i)))

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
            if 0.0 + j * step <= i < 0.0 + (j + 1) * step:
                quant.append(j)
                break

    return quant


def dequantify(quant, bits):
    """Dequantifies list of bit values
    """
    step = 1.0 / pow(2, bits)

    dequant = []
    for i in quant:
        val = ord(i) - 97
        ret = step / 2.0 + val * step
        dequant.append(ret)

    return dequant


def put_square(img, x, y, real_values, filename):
    """Generate 8x8 square into given pos image from list of 64 realValues containing numbers 0.0-1.0
       Saves under filename.
    """
    pixel_values = []
    for real_value in real_values:
        pixel_values.append(int(real_value * 255))

    idx = 0
    for a in range(8):
        for b in range(8):
            img.putpixel((x + b, y + a), pixel_values[idx])
            idx += 1
    img.save(filename, 'BMP')


def get_sequence_squares(img):
    """Get all consecutive squares from the picture
    """
    x, y = img.size

    squares = []
    for i in xrange(0, x, 8):
        for j in xrange(0, y, 8):
            squares.append(get_fixed_square(img, i, j))
    return squares


def print_picture(img, squares, name):
    """Print squares sequence into img
    """
    img = img.convert('L')
    x, y = img.size

    idx = 0
    for i in xrange(0, x, 8):
        for j in xrange(0, y, 8):
            put_square(img, i, j, squares[idx], name)
            idx += 1