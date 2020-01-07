import sys
import os
import argparse
import logging
from pathlib import Path
from extra.annotation import Formatter  # noqa

parser = argparse.ArgumentParser(description='Convert Camelyon16 xml format to'
                                 'internal json format')
parser.add_argument('xml_path', default=None, metavar='XML_PATH', type=str,
                    help='Path to the input Camelyon16 xml annotation')


def run(xml_path):
    json_path = xml_path.with_suffix(".json")
    Formatter.camelyon16xml2json(xml_path, json_path)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    xml_path = Path(args.xml_path).glob("*.xml")
    for xml in xml_path:
        run(xml)


if __name__ == '__main__':
    main()