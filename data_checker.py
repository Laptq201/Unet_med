import os
import nibabel as nib
import numpy as np
import tabulate

structure = {
    'data': {
        'training': {
            'images': {},
            'masks': {},
            'nnUNet_raw': {
                'Dataset001_BRATS': {
                    'dataset.json': None,
                    'imagesTr': {},
                    'labelsTr': {}
                }
            },
            'nnUNet_preprocessed': {},
            'nnUNet_results': {}
        },
        'test': {
            'images': {},
            'masks': {},
            'nnUNet_raw': {
                'imagesTs': {},
                'labelsTs': {}
            }
        }
    }
}


def check_structure_exists(base_path, structure):
    """
    Check if the nested folder structure exists based on the given dictionary.
    """
    for key, value in structure.items():
        current_path = os.path.join(base_path, key)
        if not os.path.exists(current_path):
            return False
        if isinstance(value, dict):
            if not check_structure_exists(current_path, value):
                return False
    return True


def create_structure(base_path, structure):
    for dir_name, children in structure.items():
        path = os.path.join(base_path, dir_name)
        os.makedirs(path, exist_ok=True)
        if children:
            create_structure(path, children)


def count_subfolders(base_path):
    num_subfolders = sum(1 for _ in os.scandir(base_path) if _.is_dir())
    return num_subfolders


def analyze_segmentation_labels(folder_path):
    """
    Analyze segmentation labels in the dataset
    """
    label_counts = {}

    # Using os.walk to iterate through files
    for root, _, files in os.walk(folder_path):
        seg_files = [os.path.join(root, file)
                     for file in files if file.endswith("-seg.nii.gz")]

        # Process segmentation files
        for file_path in seg_files:
            seg_image = nib.load(file_path)
            seg_data = seg_image.get_fdata().flatten()
            unique_labels, label_counts_per_file = np.unique(
                seg_data, return_counts=True)

            # Update the overall label counts using NumPy for efficient aggregation
            label_counts = dict(np.add.reduceat(np.array(list(label_counts.items(
            )) + list(zip(unique_labels, label_counts_per_file)), dtype=object), [0], axis=0))

    return label_counts


def main():

    INPUT_DATASET_PATH = "./input/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    base_path = os.getcwd()
    if not check_structure_exists(base_path, structure):
        create_structure(base_path, structure)
    else:
        print("The structure already exists.")

    num_subfolders = count_subfolders(INPUT_DATASET_PATH)
    print(f"Created {num_subfolders} subfolders in {INPUT_DATASET_PATH}")

    image_path = INPUT_DATASET_PATH + \
        "/BraTS-GLI-00002-000/BraTS-GLI-00002-000-t2f.nii.gz"
    test_image_obj = nib.load(image_path)
    print("Type of Image object is ", type(test_image_obj))
    test_image_data = test_image_obj.get_fdata()
    print("Type of data in Image object is ", type(test_image_data))
    print("Shape of data in Image object is ", test_image_data.shape)
    test_mask = nib.load(
        INPUT_DATASET_PATH + "/BraTS-GLI-00002-000/BraTS-GLI-00002-000-seg.nii.gz").get_fdata()
    test_mask = test_mask.astype(np.uint8)

    print("Type of data in Image object is ", type(test_mask))
    print("Shape of data in Image object is ", test_mask.shape)
    # Showing unique segmentation labels
    print("Unique Segmentation labels are ", np.unique(test_mask))

    '''
    It's kind of slow to analyze all the segmentation labels
    - Uncomment the following code to analyze the segmentation labels
    '''
    # label_counts = analyze_segmentation_labels(INPUT_DATASET_PATH)
    # total_count = sum(label_counts.values())
    # percentages = [count/total_count * 100 for count in label_counts.values()]

    # table_data = []
    # for label, count, percentage in zip(label_counts.keys(), label_counts.values(), percentages):
    #     table_data.append([label, str(count), f"{percentage:.2f}%"])

    # # Specify the table headers
    # headers = ["Label", "Count", "Percentage"]

    # # Print the table
    # print(tabulate(table_data, headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
