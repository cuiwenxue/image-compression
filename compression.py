import logging
import random
import struct

from PIL import Image

import neural_network


logger = logging.getLogger('logger')


class ZdpException(Exception):
    pass


def teach(neural_network_path, learning_image, repeat, learning_rate, hidden_layer_size=32):
    if repeat <= 0:
        raise ZdpException('Number of repetitions must be grater than 0')
    if not 0 <= learning_rate <= 1:
        raise ZdpException('Learning rate must be <0;1>')
    if not hidden_layer_size > 0 or hidden_layer_size % 8 != 0:
        raise ZdpException('Hidden layer size must be multiple of 8')
    network = neural_network.NeuralNetwork(64, [hidden_layer_size], 64, learning_rate=learning_rate)
    network.init_weights()
    logger.info('Neural network edges initialized')

    image = Image.open(learning_image)
    for i in xrange(repeat):
        x = random.randint(0, (image.size[0] - 8))
        y = random.randint(0, (image.size[1] - 8))
        colour = random.randint(0, 2)
        data = get_square(image, x, y)
        network.teach_step(data[colour], data[colour])
        logger.info('Teaching in progress... %d%%\033[F' % (100 * (i + 1) / repeat))

    logger.info('Teaching completed          ')
    neural_network.save(network, neural_network_path)
    logger.info('Neural network saved to ' + neural_network_path)


def compress(image_path, neural_network_path, compressed_image_path, bits):
    if not 1 <= bits <= 8:
        raise ZdpException('Number of bits must be <1;8>')
    network = neural_network.load(neural_network_path)

    img = Image.open(image_path)
    rgb_squares = get_sequence_squares(img)
    rgb_squares2= get_sequence_squares2(img)
    size = bits * len(network.hidden_layers[0]) / 8

    # open file and write data necessary to decompress
    file = open(compressed_image_path, 'wb')
    file.write(struct.pack('>i', img.size[0]))
    file.write(struct.pack('>i', img.size[1]))
    file.write(struct.pack('>i', len(rgb_squares)))
    file.write(struct.pack('>b', bits))
    file.write(struct.pack('>b', len(network.hidden_layers[0])))

    for i, rgb_square in enumerate(rgb_squares):
        logger.info('Compressing in progress... %d%%\033[F' % (50 * (i + 1) / len(rgb_squares)))
        for sq in rgb_square:
            network.run(sq)
            # get hidden neuron's values
            hidden_values = [neuron.value for neuron in network.hidden_layers[0]]
            quant_values = quantify(hidden_values, bits)

            # place quantified values in byte array and write to file
            bin_square = bytearray([0] * size)
            B = 0
            pos = 0
            for val in quant_values:
                bin_square[B] |= val << pos & 255
                pos += bits
                if pos == 8:
                    B += 1
                    pos = 0
                elif pos > 8:
                    B += 1
                    pos %= 8
                    bin_square[B] |= val >> (bits - pos)
            file.write(bin_square)


    for i, rgb_square in enumerate(rgb_squares2):
        logger.info('Compressing in progress... %d%%\033[F' % (50+(50 * (i + 1) / len(rgb_squares2))))
        for sq in rgb_square:
            network.run(sq)
            # get hidden neuron's values
            hidden_values = [neuron.value for neuron in network.hidden_layers[0]]
            quant_values = quantify(hidden_values, bits)

            # place quantified values in byte array and write to file
            bin_square = bytearray([0] * size)
            B = 0
            pos = 0
            for val in quant_values:
                bin_square[B] |= val << pos & 255
                pos += bits
                if pos == 8:
                    B += 1
                    pos = 0
                elif pos > 8:
                    B += 1
                    pos %= 8
                    bin_square[B] |= val >> (bits - pos)
            file.write(bin_square)

    file.close()
    logger.info('Compressing completed          ')
    logger.info('Compressed image saved to ' + compressed_image_path)


def decompress(compressed_image_path, neural_network_path, target_image_path):
    network = neural_network.load(neural_network_path)

    # read data necessary to decompress
    file = open(compressed_image_path, 'rb')
    x = struct.unpack('>i', file.read(4))[0]
    y = struct.unpack('>i', file.read(4))[0]
    number_of_rgb_squares = struct.unpack('>i', file.read(4))[0]
    bits = struct.unpack('>b', file.read(1))[0]
    hidden_layer_length = struct.unpack('>b', file.read(1))[0]

    if hidden_layer_length != len(network.hidden_layers[0]):
        raise ZdpException('Loaded network and compressed image are not compatible')

    size = bits * hidden_layer_length / 8
    mask = pow(2, bits) - 1
    squares = ([], [], [])
    logger.info('Decompressing in progress...\033[F')
    for i in xrange(number_of_rgb_squares):
        rgb_square = ([], [], [])

        # read 3 squares for each RGB colour
        for sq in rgb_square:
            B = 0
            pos = 0
            bin_square = bytearray(file.read(size))
            for j in xrange(hidden_layer_length):
                val = (bin_square[B] & (mask << pos & 255)) >> pos
                pos += bits
                if pos == 8:
                    B += 1
                    pos = 0
                elif pos > 8:
                    B += 1
                    pos %= 8
                    mask2 = pow(2, pos) - 1
                    val += (bin_square[B] & mask2) << (bits - pos)
                sq.append(val)

        # dequantify read squares, decompress and place in squares
        for i, sq in enumerate(rgb_square):
            sq = dequantify(sq, bits)
            for j, val in enumerate(sq):
                network.hidden_layers[0][j].value = val
            network.output_layer.update_values()
            output_values = [neuron.value for neuron in network.output_layer]
            squares[i].append(output_values)

    squares2 = ([], [], [])
    for i in xrange(number_of_rgb_squares):
        rgb_square = ([], [], [])

        # read 3 squares for each RGB colour
        for sq in rgb_square:
            B = 0
            pos = 0
            bin_square = bytearray(file.read(size))
            for j in xrange(hidden_layer_length):
                val = (bin_square[B] & (mask << pos & 255)) >> pos
                pos += bits
                if pos == 8:
                    B += 1
                    pos = 0
                elif pos > 8:
                    B += 1
                    pos %= 8
                    mask2 = pow(2, pos) - 1
                    val += (bin_square[B] & mask2) << (bits - pos)
                sq.append(val)

        # dequantify read squares, decompress and place in squares
        for i, sq in enumerate(rgb_square):
            sq = dequantify(sq, bits)
            for j, val in enumerate(sq):
                network.hidden_layers[0][j].value = val
            network.output_layer.update_values()
            output_values = [neuron.value for neuron in network.output_layer]
            squares2[i].append(output_values)
    print_picture(Image.new('RGB', (x, y)), squares, squares2, target_image_path)
    logger.info('Decompressing completed     ')
    logger.info('Decompressed image saved to ' + target_image_path)



def get_sequence_squares(img):
    """Get all consecutive squares from the picture.
       Pixel colour is converted to <0;1> value.
    """
    x, y = img.size
    imgsstep=8

    squares = []
    for i in xrange(0, x, imgsstep):
        for j in xrange(0, y, imgsstep):
            squares.append(get_square(img, i, j))
    return squares

def get_sequence_squares2(img):
    """Get all consecutive squares from the picture.
       Pixel colour is converted to <0;1> value.
    """
    x, y = img.size
    imgsstep=8

    squares = []
    for i in xrange(4, x+4, imgsstep):
        for j in xrange(4, y+4, imgsstep):
            squares.append(get_square(img, i, j))
    return squares


def get_square(img, x, y):
    """Get tuple of 8x8 squares for each RGB colour
       from the fixed position of the picture
    """
    img_rgb = img.convert('RGB')
    r=g=b=0
    rgb_square = ([], [], [])
    for i in range(8):
        for j in range(8):
            try:
                r, g, b = img_rgb.getpixel((x + j, y + i))
            except IndexError:
                """r=g=b=0"""
            rgb_square[0].append(r / 255.0)
            rgb_square[1].append(g / 255.0)
            rgb_square[2].append(b / 255.0)


    return rgb_square


def quantify(values, bits):
    """Quantify real numbers to the no. of bits"""
    variants = pow(2, bits)
    step = 1.0 / variants
    quant = []
    for val in values:
        for j in range(variants):
            if j * step <= val < (j + 1) * step:
                quant.append(j)
                break

    return quant


def dequantify(values, bits):
    """Dequantifies list of values - integer values to real values"""
    dequant = []
    for val in values:
        dequant.append(float(val) / (pow(2, bits) - 1))

    return dequant


def print_picture(img, squares, squares2, filename):
    """Print squares sequence into img
       Saves under filename.
    """
    x, y = img.size
    imgsstep=8
    idx = 0
    for i in xrange(0, x, imgsstep):
        for j in xrange(0, y, imgsstep):
            #if (i==0 or j==0):
            put_square(img, i, j, (squares[0][idx], squares[1][idx], squares[2][idx]))
            #else:
            #    put_square_compare(img, i, j, (squares[0][idx], squares[1][idx], squares[2][idx]),imgsstep)
            idx += 1
    improve_image(img,squares2)
    img.save(filename, 'BMP')

def put_square(img, x, y, real_values):
    """Generate 8x8 square into given pos image from list of 64 real_values containing numbers 0.0-1.0"""
    idx = 0
    for a in range(8):
        for b in range(8):
            try:
                img.putpixel((x + b, y + a), (int(real_values[0][idx] * 255), int(real_values[1][idx] * 255), int(real_values[2][idx] * 255)))
            except IndexError:
               """store previous values"""
            idx += 1

def improve_image(img,squares):
    x, y = img.size
    idx = 0
    for i in xrange(4, x+4, 8):
        for j in xrange(4, y+4, 8):
            put_square_compare(img, i, j, (squares[0][idx], squares[1][idx], squares[2][idx]))
            idx += 1

def put_square_compare(img, x, y, real_values):
    """Compare square on given pos of image with a list of 64 real_values containing numbers 0.0-1.0, and calculate avg"""
    idx = 0
    for a in range(8):
        for b in range(8):
            try:
                #print(y+a,b)
                r, g, B = img.getpixel((x + b, y + a))
                #print(r,g,b)
                w1=(abs(a-3)+abs(b-3))/8.0
                w2=1-w1
                #print(w1,w2,r)
                r=int((real_values[0][idx]* 255*w2+r*w1))#*w2+r*w1)
                if r>255:
                    r=255
                g=int((real_values[1][idx]* 255*w2+g*w1))#*w2+g*w1)
                if g>255:
                    g=255
                B=int((real_values[2][idx]* 255*w2+B*w1))#*w2+b*w1)
                if B>255:
                    B=255
                #print((x+b, y+a), img.getpixel((x + b,y + a)), (r,g,b), idx%8, idx/8, x, y, a, b)
                img.putpixel((x + b, y + a),(r,g,B))
                #print((x+b, y+a), x, y, b, a)
            except IndexError:
                """print(y+a,x+b)"""
            idx += 1