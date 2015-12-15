import argparse
import logging
import random

import neural_network


def main():
    args = parse_arguments()
    format_logger(args.debug)
    logging.getLogger('logger').info('Everything works')
    # network = neural_network.load('network.zdp')
    network = neural_network.NeuralNetwork(4, [4], 2, learning_rate=0.4)
    network.init_weights()

    for i in xrange(300000):
        a, b, c, d = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
        network.teach_step([a, b, c, d], [(a+b)/2, (c+d)/2])

    print(network.run([0.1, 0.2, 0.3, 0.15]))
    print(network.run([0.7, 0.8, 0.1, 0.9]))
    print(network.run([0.2, 0.2, 0.4, 0.6]))
    # neural_network.save(network, 'network')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Image compression using neural network')
    group_mode = parser.add_mutually_exclusive_group(required=True)
    group_mode.add_argument('-t', '--teach', dest='dir_path', type=str,
                            help='Create and teach new neural network using training data located in given directory')
    group_mode.add_argument('-c', '--compress', dest='image_path', type=str, help='Compress given image using existing neural network')
    group_mode.add_argument('-d', '--decompress', dest='image_path', type=str, help='Decompress given image using existing neural network')
    parser.add_argument('-n', dest='neural_network', type=str, help='Loads/saves neural network from given path', required=True)
    parser.add_argument('-D', '--debug', action='store_true', help='Run program in debug mode.')
    args = parser.parse_args()
    return args


def format_logger(debug=False):
    logger = logging.getLogger('logger')
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


if __name__ == '__main__':
    main()