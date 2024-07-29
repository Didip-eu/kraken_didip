# Load Segmentation object from XML

+ /kraken/lib/xml.py: XMLPage( <filename>).to_container()


# Recognize single line image

Pass a dummy segmentation object where the line boundary is the image's. Eg. in CLI interface:

+ /kraken/kraken.py: recognizer( bounds = Segmentation(type='bbox',
                                  text_direction='horizontal-lr',
                                  imagename=ctx.meta['base_image'],
                                  script_detection=False,
                                  lines=[BBoxLine(id=str(uuid.uuid4()),
                                                  bbox=(0, 0, im.size[1], im.size[0]))]) )
# Serialize to JSON

To be implemented. Should add a template in

+ /kraken/templates
