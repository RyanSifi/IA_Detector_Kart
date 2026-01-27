import os

# Create dummy annotation files for a few images in the train set
# This is just to demonstrate the format.
# The annotations will be dummy (a box in the center of the image).

train_images_path = "dataset_kart/data/train/go_kart"
image_files = os.listdir(train_images_path)

# Create a labels directory
labels_dir = "dataset_kart/data/train/labels"
os.makedirs(labels_dir, exist_ok=True)

# Create dummy annotations for the first 5 images
for i in range(min(5, len(image_files))):
    image_name = image_files[i]
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_name)
    with open(label_path, "w") as f:
        # class_id x_center y_center width height
        # Assuming class 'go_kart' is 0
        # and the object is in the center of the image
        f.write("0 0.5 0.5 0.5 0.5\n")

print("Created 5 dummy annotation files in dataset_kart/data/train/labels")

# Do the same for the validation set
val_images_path = "dataset_kart/data/val/go_kart"
image_files_val = os.listdir(val_images_path)
labels_dir_val = "dataset_kart/data/val/labels"
os.makedirs(labels_dir_val, exist_ok=True)

for i in range(min(5, len(image_files_val))):
    image_name = image_files_val[i]
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(labels_dir_val, label_name)
    with open(label_path, "w") as f:
        f.write("0 0.5 0.5 0.5 0.5\n")
print("Created 5 dummy annotation files in dataset_kart/data/val/labels")

# And for the test set
test_images_path = "dataset_kart/data/test/go_kart"
image_files_test = os.listdir(test_images_path)
labels_dir_test = "dataset_kart/data/test/labels"
os.makedirs(labels_dir_test, exist_ok=True)

for i in range(min(5, len(image_files_test))):
    image_name = image_files_test[i]
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(labels_dir_test, label_name)
    with open(label_path, "w") as f:
        f.write("0 0.5 0.5 0.5 0.5\n")
print("Created 5 dummy annotation files in dataset_kart/data/test/labels")
