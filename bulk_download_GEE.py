"""Script to download a large number of images from Google Earth Engine."""

import ee
try:
        ee.Initialize()
except Exception as e:
        ee.Authenticate()
        ee.Initialize()

def imageCollection(imagecollection, start_date, end_date, roi, bandnames):
        # Nightlights
    variable_images = ee.ImageCollection(imagecollection).filter(ee.Filter.date(start_date, end_date))

    # Get a list of all the image IDs (I'll use it to fetch images directly using ee.Image())
    image_ids = variable_images.aggregate_array("system:id").getInfo()

    #get the first image and gather bandnames
    ee.Image(image_ids[0]).bandNames().getInfo()

    #get the first image
    first_image = ee.Image(image_ids[0])

    # Boundary of South Africa
    roi = ee.Geometry.Rectangle([33.8, -0.5, 39.8, 5.0]) #xmin, ymin, xmax, ymax

    # Function to mask all values outside the geometry
    def maskOutside(image, geometry):
        mask = ee.Image.constant(1).clip(geometry).mask() # add .not() to mask inside
        return image.updateMask(mask)
    

    """Incomplete!!!"""