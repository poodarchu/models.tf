"""
    Extract background images from a tar archive
"""

__all__ = {
    'extract_backgrounds',
}

import os
import sys
import tarfile

import cv2
import numpy as np

def img_from_file(file):
    a = np.asarray(bytearray(file.read()), dtype=np.uint8)
    return cv2.imdecode(a, cv2.IMREAD_GRAYSCALE)

"""
    Extract backgrounds from provided tar archive.
    
    JPEGs from the archive are converted into grayscale, and cropped/resized to
    256x256, and saved in ./bgs/.
"""
def extract_bgs(archive_name):
    os.mkdir('bgs')
    t = tarfile.open(name=archive_name)

    def members():
        m = t.next()
        while m:
            yield m
            m = t.next()

    index = 0

    for m in members():
        if not m.name.endswith('.jpg'):
            continue
        f = t.extractfile(m)
        try:
            img = img_from_file(f)
        finally:
            f.close()
        if img is None:
            continue

        if img.shape[0] > img.shape[1]:
            img = img[:img.shape[1], :]
        else:
            img = img[:, :img.shape[0]]

        if img.shape[0] > 256:
            img = cv2.resize(img, (256, 256))

        fname = "bgs/{.08}.jpg".format(index)
        print fname

        rc = cv2.imwrite(fname, img)

        if not rc:
            raise Exception("Failed to write file {}".format(fname))

        index += 1

if __name__ == "__main__":
    extract_bgs(sys.argv[1])


