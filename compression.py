import logging
import random
import struct

from PIL import Image

import neural_network


def teach(neural_network_path, learning_image, repeat, learning_rate):
    network = neural_network.NeuralNetwork(64, [32], 64, learning_rate=learning_rate)
    network.init_weights()
    logging.getLogger('logger').info('Neural network edges initialized')

    image = Image.open(learning_image)
    for i in xrange(repeat):
        data = get_random_square(image)
        network.teach_step(data, data)
        logging.getLogger('logger').info('Teaching in progress... %d%%\033[F' % (100 * (i + 1) / repeat))

    logging.getLogger('logger').info('Teaching completed          ')
    neural_network.save(network, neural_network_path)
    logging.getLogger('logger').info('Neural network saved to ' + neural_network_path)


def compress(image_path, neural_network_path, compressed_image_path, bits):
    network = neural_network.load(neural_network_path)

    img = Image.open(image_path)
    rgb_squares = get_sequence_squares(img)

    file = open(compressed_image_path, 'wb')
    file.write(struct.pack('>i', img.size[0]))
    file.write(struct.pack('>i', img.size[1]))
    file.write(struct.pack('>i', len(rgb_squares)))
    file.write(struct.pack('>b', bits))
    file.write(struct.pack('>b', len(network.hidden_layers[0])))

    for i, rgb_square in enumerate(rgb_squares):
        logging.getLogger('logger').info('Compressing in progress... %d%%\033[F' % (100 * (i + 1) / len(rgb_squares)))
        for sq in rgb_square:
            network.run(sq)
            hidden_values = [neuron.value for neuron in network.hidden_layers[0]]
            quant_values = quantify(hidden_values, bits)

            size = bits * len(network.hidden_layers[0]) / 8
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
    logging.getLogger('logger').info('Compressing completed          ')
    logging.getLogger('logger').info('Compressed image saved to ' + compressed_image_path)


def decompress(compressed_image_path, neural_network_path, target_image_path):
    logging.getLogger('logger').info('Compressing in progress...')
    network = neural_network.load(neural_network_path)

    file = open(compressed_image_path, 'rb')

    x = struct.unpack('>i', file.read(4))[0]
    y = struct.unpack('>i', file.read(4))[0]
    number_of_rgb_squares = struct.unpack('>i', file.read(4))[0]
    bits = struct.unpack('>b', file.read(1))[0]
    hidden_layer_length = struct.unpack('>b', file.read(1))[0]
    size = bits * hidden_layer_length / 8
    mask = pow(2, bits) - 1

    rgb_squares = []
    for i in xrange(number_of_rgb_squares):
        rgb_square = ([], [], [])

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
        rgb_squares.append(rgb_square)

    squares = ([], [], [])
    for rgb_square in rgb_squares:
        for i, sq in enumerate(rgb_square):
            for j, val in enumerate(sq):
                network.hidden_layers[0][j].value = float(val) / (pow(2, bits) - 1)
            network.output_layer.update_values()
            output_values = [neuron.value for neuron in network.output_layer]
            squares[i].append(output_values)

    print_picture(Image.new('RGB', (x, y)), squares, target_image_path)
    logging.getLogger('logger').info('Decompressing completed')
    logging.getLogger('logger').info('Decompressed image saved to ' + target_image_path)


def new_image(size):
    return Image.new('F', size)


def get_random_square(img):
    """Get random square 8x8 from the grayscale picture
       and quantify their colours to <0;1>
    """
    img_grayscale = img.convert('L')
    x = random.randint(0, (img_grayscale.size[0] - 8))
    y = random.randint(0, (img_grayscale.size[1] - 8))

    square = []
    for i in range(8):
        for j in range(8):
            square.append(img_grayscale.getpixel((x + j, y + i)))

    return dequantify(square, 8)


def get_sequence_squares(img):
    """Get all consecutive squares from the picture
    """
    x, y = img.size

    squares = []
    for i in xrange(0, x, 8):
        for j in xrange(0, y, 8):
            squares.append(get_fixed_square(img, i, j))
    return squares


def get_fixed_square(img, x, y):
    """Get tuple of 8x8 squares for each RGB colour
       from the fixed position of the picture
    """
    img_rgb = img.convert('RGB')
    rgb_square = ([], [], [])

    for i in range(8):
        for j in range(8):
            r, g, b = img_rgb.getpixel((x + j, y + i))
            rgb_square[0].append(r / 255.0)
            rgb_square[1].append(g / 255.0)
            rgb_square[2].append(b / 255.0)

    return rgb_square


def quantify(values, bits):
    """Quantify real number output of layer to the no. of bits
    """
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
    """Dequantifies list of values
    """
    dequant = []
    for val in values:
        dequant.append(float(val) / (pow(2, bits) - 1))

    return dequant


def print_picture(img, squares, name):
    """Print squares sequence into img
    """
    x, y = img.size

    idx = 0
    for i in xrange(0, x, 8):
        for j in xrange(0, y, 8):
            put_square(img, i, j, (squares[0][idx], squares[1][idx], squares[2][idx]), name)
            idx += 1


def put_square(img, x, y, real_values, filename):
    """Generate 8x8 square into given pos image from list of 64 realValues containing numbers 0.0-1.0
       Saves under filename.
    """
    idx = 0
    for a in range(8):
        for b in range(8):
            img.putpixel((x + b, y + a), (int(real_values[0][idx] * 255), int(real_values[1][idx] * 255), int(real_values[2][idx] * 255)))
            idx += 1
    img.save(filename, 'BMP')