import argparse
import logging


def main():
    args = parse_arguments()
    format_logger(args.debug)
    logging.getLogger('logger').info('Everything works.')


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
    logger.addHandler(console_handler)


if __name__ == '__main__':
    main()