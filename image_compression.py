import argparse
import logging
import sys
import random
import compression
import pickle

import neural_network


def teach(repeat, learning_rate, neural_network_path):
    logging.getLogger('logger').info('Running program in teaching mode')
    network = neural_network.NeuralNetwork(64, [16], 64, learning_rate=learning_rate)
    network.init_weights()
    logging.getLogger('logger').info('Neural network edges initialized')

    sys.stdout.write('Teaching in progress    ')
    for i in xrange(repeat):
        sys.stdout.write('\b\b\b')
        # TODO teaching algorithm implementation
        data = compression.getRandomSquare(compression.openImage('59.bmp'))
        network.teach_step(data, data)
        sys.stdout.write("%d%%" % (100 * (i + 1) / repeat))
        sys.stdout.flush()
        sys.stdout.write('\n')

    print "\n\tTEST RUN\n"
    data = compression.getRandomSquare(compression.openImage('59.bmp'))
    print data
    print (network.run(data))

    logging.getLogger('logger').info('Teaching completed')
    neural_network.save(network, neural_network_path)
    logging.getLogger('logger').info('Neural network saved to ' + neural_network_path)


def compress(image_path, neural_network_path):
    logging.getLogger('logger').info('Running program in compression mode')
    try:
        network = neural_network.load(neural_network_path)
    except neural_network.NeuralNetworkException as exc:
        logging.getLogger('logger').critical('Cannot load neural network: ' + exc.message)
        exit(-1)
    # TODO compression algorithm
    img = compression.openImage(image_path)
    squares = compression.getSequenceSquares(img)
    ret = []
    hidden = []
    hidden.append(img.size)
    print squares[0]
    for sq in squares:
        ret.append(network.run(sq))
        hidden.append(compression.quantify(network.hidden_layers[0], 8))
    compression.printPicture(img, ret, 'afterRun.bmp')
    pickle.dump(hidden, open('cmpIMG' + '.zdp', 'wb'), pickle.HIGHEST_PROTOCOL)


def decompress(image_path, neural_network_path):
    logging.getLogger('logger').info('Running program in decompression mode')
    try:
        network = neural_network.load(neural_network_path)
    except neural_network.NeuralNetworkException as exc:
        logging.getLogger('logger').critical('Cannot load neural network: ' + exc.message)
        exit(-1)
        # TODO decompression algorithm
    img = compression.openImage(image_path)
    try:
        hidden = pickle.load(open('cmpIMG.zdp', 'rb'))
    except:
        logging.getLogger('logger').critical('Cannot load compressed archive: ' + exc.message)



def main():
    args = parse_arguments()
    format_logger(args.debug)
    if args.teach:
        # TODO get those parameters from argparse
        teach(100000, 0.5, args.neural_network)
    elif args.compress:
        compress(args.compress, args.neural_network)
    else:
        decompress(args.decompress, args.neural_network)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Image compression using neural network')
    group_mode = parser.add_mutually_exclusive_group(required=True)
    group_mode.add_argument('-t', '--teach', type=str,
                            help='Create and teach new neural network using training data located in given directory')
    group_mode.add_argument('-c', '--compress', type=str, help='Compress given image using existing neural network')
    group_mode.add_argument('-d', '--decompress', type=str, help='Decompress given image using existing neural network')
    parser.add_argument('-n', dest='neural_network', type=str, help='Loads/saves neural network from given path', required=True)
    parser.add_argument('-D', '--debug', action='store_true', help='Run program in debug mode.')
    args = parser.parse_args()
    return args


def format_logger(debug=False):
    logger = logging.getLogger('logger')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger('logger').info('Running program in debug mode')
    else:
        logger.setLevel(logging.INFO)


if __name__ == '__main__':
    main()