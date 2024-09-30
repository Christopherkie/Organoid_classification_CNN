from PIL import Image
import os

folder = r'training_dataset/p3_bad_aug'
new = "training_dataset/p3_bad_jpg/"
for root, dirs, files in os.walk(folder, topdown=False):
    for name in files:
        print(os.path.join(root, name))
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".tif":
            if os.path.isfile(os.path.splitext(os.path.join(new, name))[0] + ".jpg"):
                print("A jpeg file already exists for %s" % name)
            # If a jpeg is *NOT* present, create one from the tif.
            else:
                outfile = os.path.splitext(os.path.join(name))[0] + ".jpg"
                try:
                    im = Image.open(os.path.join(root, name))
                    print("Generating jpeg for %s" % name)
                    im.thumbnail(im.size)
                    im.save("training_dataset/p3_bad_jpg/" + outfile, "JPEG", quality=100)
                except Exception as e:
                    print(e)
