# Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)
# ALtered by: Joseph Rowell (ucabcrr@ucl.ac.uk)

# USAGE 
# # Navigate to code file 
# !python database.py  --database_path $DATABASE_PATH

import sys
import sqlite3 
import numpy as np


IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.full(4, np.NaN), prior_t=np.full(3, np.NaN), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3),
                              qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                              tvec=np.zeros(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H),
             array_to_blob(qvec), array_to_blob(tvec)))


def example_usage():
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    if os.path.exists(args.database_path):
        print("ERROR: database path already exists -- will not modify it.")
        return

    # Open the database.

    db = COLMAPDatabase.connect(args.database_path)

    # For convenience, try creating all the tables upfront.

    db.create_tables()

    # Create dummy cameras.

    model1, width1, height1, params1 = \
        0, 1024, 768, np.array((1024., 512., 384.))
    model2, width2, height2, params2 = \
        2, 1024, 768, np.array((1024., 512., 384., 0.1))

    camera_id1 = db.add_camera(model1, width1, height1, params1)
    camera_id2 = db.add_camera(model2, width2, height2, params2)

    # Create dummy images.

    image_id1 = db.add_image("image1.png", camera_id1)
    image_id2 = db.add_image("image2.png", camera_id1)
    image_id3 = db.add_image("image3.png", camera_id2)
    image_id4 = db.add_image("image4.png", camera_id2)

    # Create dummy keypoints.
    #
    # Note that COLMAP supports:
    #      - 2D keypoints: (x, y)
    #      - 4D keypoints: (x, y, theta, scale)
    #      - 6D affine keypoints: (x, y, a_11, a_12, a_21, a_22)

    num_keypoints = 1000
    keypoints1 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints2 = np.random.rand(num_keypoints, 2) * (width1, height1)
    keypoints3 = np.random.rand(num_keypoints, 2) * (width2, height2)
    keypoints4 = np.random.rand(num_keypoints, 2) * (width2, height2)

    db.add_keypoints(image_id1, keypoints1)
    db.add_keypoints(image_id2, keypoints2)
    db.add_keypoints(image_id3, keypoints3)
    db.add_keypoints(image_id4, keypoints4)

    # Create dummy matches.

    M = 50
    matches12 = np.random.randint(num_keypoints, size=(M, 2))
    matches23 = np.random.randint(num_keypoints, size=(M, 2))
    matches34 = np.random.randint(num_keypoints, size=(M, 2))

    db.add_matches(image_id1, image_id2, matches12)
    db.add_matches(image_id2, image_id3, matches23)
    db.add_matches(image_id3, image_id4, matches34)

    # Commit the data to the file.

    db.commit()

    # Read and check cameras.

    rows = db.execute("SELECT * FROM cameras")

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id1
    assert model == model1 and width == width1 and height == height1
    assert np.allclose(params, params1)

    camera_id, model, width, height, params, prior = next(rows)
    params = blob_to_array(params, np.float64)
    assert camera_id == camera_id2
    assert model == model2 and width == width2 and height == height2
    assert np.allclose(params, params2)

    # Read and check keypoints.
 
    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints"))

    assert np.allclose(keypoints[image_id1], keypoints1)
    assert np.allclose(keypoints[image_id2], keypoints2)
    assert np.allclose(keypoints[image_id3], keypoints3)
    assert np.allclose(keypoints[image_id4], keypoints4)

    # Read and check matches.

    pair_ids = [image_ids_to_pair_id(*pair) for pair in
                ((image_id1, image_id2),
                 (image_id2, image_id3),
                 (image_id3, image_id4))]

    matches = dict(
        (pair_id_to_image_ids(pair_id),
         blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in db.execute("SELECT pair_id, data FROM matches")
    )

    assert np.all(matches[(image_id1, image_id2)] == matches12)
    assert np.all(matches[(image_id2, image_id3)] == matches23)
    assert np.all(matches[(image_id3, image_id4)] == matches34)

    # Clean up.

    db.close()

    if os.path.exists(args.database_path):
        os.remove(args.database_path)

def add_column_in_csv(input_file, output_file, transform_row):
    """ Append a column in existing csv using csv.reader / csv.writer classes"""
    from csv import writer
    from csv import reader
    # Open the input_file in read mode and output_file in write mode
    with open(input_file, 'r') as read_obj, \
            open(output_file, 'w') as write_obj:
        # Create a csv.reader object from the input file object
        csv_reader = reader(read_obj)
        # Create a csv.writer object from the output file object
        csv_writer = writer(write_obj)
        # Read each row of the input csv file as list
        for row in csv_reader:
            # Pass the list / row in the transform function to add column text for this row
            transform_row(row, csv_reader.line_num)
            # Write the updated row / list to the output file
            csv_writer.writerow(row)
    
def extract_keypoints(filename, LABEL_NAMES):
    '''
    Function to extract SIFT keypoints from colmap database sparse reconstruction
    INPUT: Path to CSV
    OUTPUT: CSV file with imageId, Keypoints X, Keypoints Y, Semantic Label (blank)
    
    '''
    import os
    import argparse
    import csv
    import pandas as pd

    #Avoid np array truncation, comment this out if debugging
    np.set_printoptions(threshold=sys.maxsize) 

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    # if os.path.exists(args.database_path):
    #     print("Database path exists.")
        

    # Open the database.

    db = COLMAPDatabase.connect(args.database_path)

    # Read and check keypoints.
 
    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 6)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints"))
    #print(keypoints[1])
    #assert np.allclose(keypoints[image_id1], keypoints1)

    # write to csv
    with open(filename, 'w') as f:
        for key in keypoints.keys():
            for i in range(len(keypoints[key])):
                # imageId, X, Y, Semantic LabeL
                intensity = get_pixel_intensity(key,keypoints[key][i,:][0],keypoints[key][i,:][1])
                semanticLabel = get_label_from_intensity(intensity, LABEL_NAMES)
                f.write("%s, %s, %s ,%s, %s\n" % (key, keypoints[key][i,:][0], keypoints[key][i,:][1], intensity, semanticLabel))
                #print("%s, %s, %s ,%s, %s\n" % (key, keypoints[key][i,:][0], keypoints[key][i,:][1], intensity, semanticLabel))

                #f.write("%s, %s, %s, %s\n" % (key, keypoints[key][i,:][0], keypoints[key][i,:][1], None))
                #print(key, keypoints[key][:][0],keypoints[key][:][1])

    assert os.path.exists(filename)

    if os.path.exists(filename):
        print("Written keypoints to .csv successfully.")
    #np.savetxt('keypoints.csv', keypoints, delimiter=',')

    db.close()

    file = pd.read_csv(filename)
    # adding headers
    headerlist = ["imageId", "keypoints X", "keypoints Y", "pixel intensity", "semantic label"]

    # converting data frame to csv
    file.to_csv(filename, header=headerlist, index=False)



def get_pixel_intensity(imageId, X, Y):
    '''
    Function to determine semantic label of colmap identified keypoints using DeepLab 
    semantic segmentation as a reference
    INPUT: csvPath path to csv file that holds keypoint location data
    OUTPUT: append a column of semantic label on to csv file
    '''
    from PIL import Image
    # import pandas as pd
    # import csv
    # import os

    # #load in csv file
    # csvfile = open(csvPath)
    # csvreader = csv.reader(csvfile)
    # rows = []
    # for row in csvreader:
    #     rows.append(row)

    # # create image object (iterate through file eventually)
    # for images in os.listdir("/home/joerowelll/COMP0132/brighton_workspace/segmentation"):
    # # check if the image ends with png
    #     if (images.endswith(".png")):
    #         #print(images)
    # for i in range(len(row[:][0])):
    #     im = Image.open(r"/home/joerowelll/COMP0132/brighton_workspace/segmentation/out" + row[i][0] +".png")
    #     #im = Image.open(imagesPath)
    
    #     px = im.load()
    #     print(rows[1][1])
    #     print(rows[1][2])
    #     coordinate = (int(float(rows[1][1])),int(float(row[1][2]))) # change the 1 to an iterative Why is coordinate decimal?

    #     semLabel = im.getpixel(coordinate)
    #     print(semLabel)
        
    #     add_column_in_csv(csvPath, 'output_3.csv', lambda row, line_num: row.append(semLabel))
    im = Image.open(r"/home/joerowelll/COMP0132/brighton_workspace/segmentation/out" + str(imageId) +".png")
    px = im.load()
    #print(im.width, im.height)
    
    coordinate = (int(float(X)/1.055), int(float(Y)/1.055)) #convert from string to float, then round as SIFT features have subpixel accuracy.
    #scvaled down 4% diagonally as colmap scales images.

    # if (coordinate[0] > im.width):
    #     coordinate = (im.width/1.01, coordinate[1]) 
    # if (coordinate[1] > im.height):
    #     coordinate = (coordinate[0], im.height/1.01) 
    # Coordinate my be larger than image by 1% diagonally 
     
    #print(coordinate)
    pixelIntensity = im.getpixel(coordinate)

    return pixelIntensity

def get_label_from_intensity(intensity, LABEL_NAMES):
    return LABEL_NAMES[intensity]


def create_cityscapes_label_grayscalemap():
    """Creates a label grayscalemap used in CITYSCAPES segmentation benchmark.
    Returns:
    A grayscale map for getting labels in semantic labelling pixel intensity
    """
    
    colormap = np.zeros((256, 1), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap



if __name__ == "__main__":
    #Pascal VOC
    # LABEL_NAMES = np.asarray([
    #     'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    #     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    #     'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    # ])

    #Cityscapes
    # LABEL_NAMES = np.asarray([
    #     'unlabeled', 'ego vehicle', 'out of roi', 'static', 'dynamic', 'ground', 'road',
    #     'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge',
    #     'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    #     'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 
    #     'motorcycle', 'bicycle', 'license plate'
    # ])

    ##ADE20K
    LABEL_NAMES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation',  'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 
    'motorcycle', 'bicycle'
    ])

    extract_keypoints("/home/joerowelll/COMP0132/brighton_workspace/brightonKeypoints.csv", LABEL_NAMES)
    
# EXAMPLE USAGE 
#$ python database.py  --database_path ~/COMP0132/brighton_workspace/database.db