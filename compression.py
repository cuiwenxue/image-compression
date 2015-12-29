import logging
import neural_network
import random

from PIL import Image


def teach(neural_network_path, learning_image, repeat, learning_rate):
    network = neural_network.NeuralNetwork(64, [32], 64, learning_rate=learning_rate)
    network.init_weights()
    logging.getLogger('logger').info('Neural network edges initialized')

    image = open_image(learning_image)
    for i in xrange(repeat):
        data = get_random_square(image)
        network.teach_step(data, data)
        logging.getLogger('logger').info('Teaching in progress... %d%%\033[F' % (100 * (i + 1) / repeat))

    logging.getLogger('logger').info('Teaching completed          ')
    neural_network.save(network, neural_network_path)
    logging.getLogger('logger').info('Neural network saved to ' + neural_network_path)


def compress(image_path, neural_network_path, compressed_image_path, bits):
    network = neural_network.load(neural_network_path)

    img = open_image(image_path)
    squares = get_sequence_squares(img)

    file = open(compressed_image_path, 'w')
    file.write(str(img.size[0]) + ' ' + str(img.size[1]) + ' ' + str(bits) + '\n')
    for sq in squares:
        network.run(sq)
        hidden_values = [neuron.value for neuron in network.hidden_layers[0]]
        quant_values = quantify(hidden_values, bits)
        for val in quant_values:
            x = val + 97
            file.write(chr(x))
        file.write('\n')


def decompress(compressed_image_path, neural_network_path, target_image_path):
    network = neural_network.load(neural_network_path)

    file = open(compressed_image_path, 'r')
    x, y, bits = file.readline().split()

    quant_values = []
    dequant_values = []
    quant_line = []

    img = new_image((int(x), int(y)))
    for line in file:
        for c in line:
            if c != '\n':
                quant_line.append(c)
        quant_values.append(quant_line)
        quant_line = []

    for i in quant_values:
        dequant_values.append(dequantify(i, int(bits)))

    squares = []
    for i in xrange(len(dequant_values)):
        for j, val in enumerate(dequant_values[i], start=0):
            network.hidden_layers[0][j].value = val
        network.output_layer.update_values()
        output_values = [neuron.value for neuron in network.output_layer]
        squares.append(output_values)

    print_picture(img, squares, target_image_path)


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