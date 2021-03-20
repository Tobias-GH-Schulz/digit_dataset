import splitfolders  # or import split_folders

input_folder = "/Users/tobiasschulz/Documents/GitHub/digit_dataset/digit_from_grid/img_data/"
output_folder = "/Users/tobiasschulz/Documents/GitHub/digit_dataset/digit_from_grid/img_splitted/"


# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.7, .15, .15), group_prefix=None) # default values

# Split val/test with a fixed number of items e.g. 100 for each set.
# To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
#splitfolders.fixed("input_folder", output="output", seed=1337, fixed=(100, 100), oversample=False, group_prefix=None) # default values