import errno
import os
from lxml import etree


def ensure_dir(path):
    """Ensure that the directory specified exists, and if not, create it."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:  # File exists already when multiple python process require the same dir
            raise
    return path


def make_xml(name, nsl):
    """Create an xml file."""
    xsi = "http://www.w3.org/2001/XMLSchema-instance"
    ns = {"xsi": xsi}
    attr = {"{%s}noNamespaceSchemaLocation" % xsi: nsl}
    t = etree.Element(name, attrib=attr, nsmap=ns)
    return t


def print_xml(content, dir_file):
    """Print information from a dict into an xml file."""
    etree.ElementTree(content).write(
        dir_file, pretty_print=True, encoding='UTF-8', xml_declaration=True)

