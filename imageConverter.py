from pathlib import Path
import imghdr
import cv2

data_dir = "test_keep"
image_extensions = [".png", ".jpg"]  # add there all your images file extensions

bad_images=[]

img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
for filepath in Path(data_dir).rglob("*"):
    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)
        is_jfif = False
        try:
            print(filepath)
            img = cv2.imread(filepath)
            shape = img.shape
            fobj = open(filepath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        except Exception as e:
            print('file ', filepath, ' is not a valid image file')
            print(e)
            bad_images.append(filepath)
        finally:
            fobj.close()

        if not is_jfif:
            print(f"{filepath} is not JFIF")
        
        if img_type is None:
            print(f"{filepath} is not an image")
        elif img_type not in img_type_accepted_by_tf:
            print(f"{filepath} is a {img_type}, not accepted by TensorFlow")

print(bad_images)