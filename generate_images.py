import os
import xml.etree.ElementTree as ET

def extract_kanji_data(kanjidic2_xml_path, output_file):
    # Parse the Kanjidic2 XML file
    tree = ET.parse(kanjidic2_xml_path)
    root = tree.getroot()

    # Open the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Iterate through each <character> element
        for character in root.findall('.//character'):
            literal = character.find('literal').text
            if not literal:
                continue

            # Extract content from <meaning> tags
            meanings = [meaning.text for meaning in character.findall('.//meaning')]
            # Extract content from <reading> tags
            readings = [reading.text for reading in character.findall('.//reading')]

            # Write to the output file
            f.write(f"Character: {literal}\n")
            f.write(f"Meanings: {', '.join(meanings)}\n")
            f.write(f"Readings: {', '.join(readings)}\n")
            f.write("\n")

if __name__ == "__main__":
    KANJIDIC2_XML_PATH = 'kanjidic2.xml'  # Replace with the path to your kanjidic2.xml file
    OUTPUT_FILE = 'data/kanji_data.txt'  # Path to the output text file

    # Extract kanji data
    extract_kanji_data(KANJIDIC2_XML_PATH, OUTPUT_FILE)