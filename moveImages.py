import os
import shutil

def move_images(source_dirs, target_base_dir):
    for source_dir in source_dirs:
        for bird_name in os.listdir(source_dir):
            source_subdir = os.path.join(source_dir, bird_name)
            if os.path.isdir(source_subdir):
                target_subdir = os.path.join(target_base_dir, bird_name)
                os.makedirs(target_subdir, exist_ok=True)

                for filename in os.listdir(source_subdir):
                    source_file = os.path.join(source_subdir, filename)
                    if os.path.isfile(source_file):
                        target_file = os.path.join(target_subdir, filename)
                        shutil.move(source_file, target_file)
                        print(f"Moved {source_file} to {target_file}")

if __name__ == "__main__":
    # Define the source directories
    source_dirs = [
        os.path.join(os.getcwd(), 'test_keep'),
        os.path.join(os.getcwd(), 'train_keep'),
        os.path.join(os.getcwd(), 'valid_keep')
    ]

    # Define the target base directory
    target_base_dir = os.path.join(os.getcwd(), 'training_images')

    # Move the images
    move_images(source_dirs, target_base_dir)

    print("Image moving complete!")
