# External imports
import numpy as np

def binary_list_to_string(binary_list, num_bits=6, offset=48):
    if len(binary_list) % num_bits != 0:
        raise ValueError("The binary list length must be a multiple of 8.")
    binary_list = [int(round(b)) for b in binary_list] 
    chars = []
    for i in range(0, len(binary_list), num_bits):
        byte = binary_list[i : i + num_bits]
        byte_as_int = offset + int("".join(map(str, byte)), 2)
        chars.append(chr(byte_as_int))
    return "".join(chars)

def array_to_string(arr: np.array, num_bits=6, offset=48):
    raveled = list(arr.ravel())
    if len(raveled) % num_bits != 0:
        padding = num_bits - (len(raveled) % num_bits)
        raveled.extend([0] * padding)
    result = binary_list_to_string(raveled, num_bits, offset)
    return result

def generate_submission_file(predictions, output_dir):
    mask_names = ["rg20090520_scan.png.ppm", "rg20090121_scan.png.ppm"]
    with open(f"{output_dir}/submission.csv", "w") as f:
        f.write("Id,Target\n")

        # Iterate over the predictions for each image
        for mask_id in range(len(predictions)):
            prediction = predictions[mask_id]
            
            # Iterate over the rows of the prediction and write them in the required format
            for idx_row, row in enumerate(prediction):
                mystr = array_to_string(row)
                f.write(f"{mask_names[mask_id]}_{idx_row},\"{mystr}\"\n")