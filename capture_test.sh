#!/bin/bash

# Create the directory to save images if it doesn't exist
output_dir="test_images"
mkdir -p "$output_dir"

# Base command and file path
base_command="libcamera-still --autofocus-mode manual --tuning-file /usr/share/libcamera/ipa/rpi/vc4/ov5647_af.json"

# Loop through lens positions from 0 to 10 in increments of 0.05
for lens_pos in $(seq 0 0.05 10); do
    # Format the lens position for the filename
    formatted_lens_pos=$(printf "%.2f" $lens_pos)
    
    # Set the output filename
    output_file="$output_dir/image_lens_$formatted_lens_pos.jpg"
    
    # Complete the command with the current lens position and output file
    command="$base_command --lens-position $lens_pos -o $output_file"
    
    # Run the command
    echo "Capturing image with lens position $formatted_lens_pos..."
    eval $command
done

echo "Image capture completed."

