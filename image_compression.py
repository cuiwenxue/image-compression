import argparse
import logging

import compression
import neural_network


def teach(repeat, learning_rate, neural_network_path, learning_image):
    logging.getLogger('logger').info('Running program in teaching mode')
    network = neural_network.NeuralNetwork(64, [32], 64, learning_rate=learning_rate)
    network.init_weights()
    logging.getLogger('logger').info('Neural network edges initialized')

    image = compression.open_image(learning_image)
    for i in xrange(repeat):
        data = compression.get_random_square(image)
        network.teach_step(data, data)
        logging.getLogger('logger').info('Teaching in progress... %d%%\033[F' % (100 * (i + 1) / repeat))

    logging.getLogger('logger').info('Teaching completed          ')
    neural_network.save(network, neural_network_path + '.mkm')
    logging.getLogger('logger').info('Neural network saved to ' + neural_network_path + '.mkm')


def compress(image_path, neural_network_path, bits, compressed_image_path):
    logging.getLogger('logger').info('Running program in compression mode')
    try:
        network = neural_network.load(neural_network_path)
    except neural_network.NeuralNetworkException as exc:
        logging.getLogger('logger').critical('Cannot load neural network: ' + exc.message)
        exit(1)
    except IOError as exc:
        logging.getLogger('logger').critical('Cannot load neural network: ' + exc.strerror)
        exit(exc.errno)

    img = compression.open_image(image_path)
    squares = compression.get_sequence_squares(img)

    file = open(compressed_image_path + '.zdp', 'w')
    file.write(str(img.size[0]) + ' ' + str(img.size[1]) + ' ' + str(bits) + '\n')
    for sq in squares:
        network.run(sq)
        hidden_values = [neuron.value for neuron in network.hidden_layers[0]]
        quant_values = compression.quantify(hidden_values, bits)
        for val in quant_values:
            x = val + 97
            file.write(chr(x))
        file.write('\n')


def decompress(compressed_image_path, neural_network_path, target_image_path):
    logging.getLogger('logger').info('Running program in decompression mode')
    try:
        network = neural_network.load(neural_network_path)
    except neural_network.NeuralNetworkException as exc:
        logging.getLogger('logger').critical('Cannot load neural network: ' + exc.message)
        exit(1)
    except IOError as exc:
        logging.getLogger('logger').critical('Cannot load neural network: ' + exc.strerror)
        exit(exc.errno)

    file = open(compressed_image_path, 'r')
    x, y, bits = file.readline().split()

    quant_values = []
    dequant_values = []
    quant_line = []

    img = compression.new_image((int(x), int(y)))
    for line in file:
        for c in line:
            if c != '\n':
                quant_line.append(c)
        quant_values.append(quant_line)
        quant_line = []

    for i in quant_values:
        dequant_values.append(compression.dequantify(i, int(bits)))

    squares = []
    for i in xrange(len(dequant_values)):
        for j, val in enumerate(dequant_values[i], start=0):
            network.hidden_layers[0][j].value = val
        network.output_layer.update_values()
        output_values = [neuron.value for neuron in network.output_layer]
        squares.append(output_values)

    compression.print_picture(img, squares, target_image_path)


def main():
    args = parse_arguments()
    format_logger()

    if args.gui:
        # TODO run in graphical mode
        raise NotImplementedError()
    if args.teach:
        teach(args.teach[0], args.teach[1], args.teach[2], args.teach[3])
    elif args.compress:
        compress(args.compress[0], args.compress[1], args.compress[2], args.compress[3])
    else:
        decompress(args.decompress[0], args.decompress[1], args.decompress[2])


class TeachAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        repeat, rate, network, train = values
        if not repeat.isdigit():
            parser.error('REPEAT value must by positive integer')

        if not 0 <= float(rate) <= 1:
            parser.error('RATE value must [0.0; 1.0]')

        setattr(namespace, self.dest, [int(values[0]), float(values[1]), values[2], values[3]])


class CompressAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        image, network, bit, compressed = values
        if not bit.isdigit():
            parser.error('BIT value must by positive integer')

        setattr(namespace, self.dest, [values[0], values[1], int(values[2]), values[3]])


def parse_arguments():
    parser = argparse.ArgumentParser(description='Image compression using neural network', formatter_class=argparse.RawTextHelpFormatter)
    group_mode = parser.add_mutually_exclusive_group(required=True)
    group_mode.add_argument('-g', '--gui', action='store_true', help='Run program in graphical user interface')
    group_mode.add_argument('-t', dest='teach', nargs=4, metavar=('REPEAT', 'RATE', 'NETWORK', 'TRAIN'), action=TeachAction,
                            help='Create and teach new neural network using training da-\n'
                                 'ta located in given directory\n\n'
                                 'REPEAT   indicates how  many steps has to be taken dur-\n'
                                 '         ing teaching algorithms\n'
                                 'RATE     indicates learning rate, speed of teaching al-\n'
                                 '         gorithm\n'
                                 'NETWORK  indicates path where  generated neural network\n'
                                 '         network will be saved\n'
                                 'TRAIN    indicates path where is located training image\n')
    group_mode.add_argument('-c', dest='compress', nargs=4, metavar=('IMAGE', 'NETWORK', 'BIT', 'COMPRESSED'), action=CompressAction,
                            help='Compress given image using existing neural network\n\n'
                                 'IMAGE    indicates path to compress image\n'
                                 'NETWORK  indicates path where neural network is located\n'
                                 'BIT      indicates  number of  bits  per pixel  in com-\n'
                                 '         pressed image'
                                 'COMPRESSED indicates path  where compressed  image will\n'
                                 '           be saved\n', )
    group_mode.add_argument('-d', dest='decompress', nargs=3, metavar=('COMPRESSED', 'NETWORK', 'IMAGE'),
                            help='Decompress given image using existing neural network\n\n'
                                 'COMPRESSED indicates path to compressed image\n'
                                 'NETWORK  indicates path where neural network is located\n'
                                 'IMAGE    indicates path  where  decompressed image will\n'
                                 '         be saved\n', )
    args = parser.parse_args()
    return args


def format_logger():
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


if __name__ == '__main__':
    main()