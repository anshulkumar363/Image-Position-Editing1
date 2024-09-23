from src import ImageEditor
import argparse
from pathlib import Path

lama_config = str(Path(__file__).resolve().parent / "lama/configs/prediction/default.yaml")
lama_ckpt = str(Path(__file__).resolve().parent / "lama/pretrained-models/big-lama")

def main():
    parser = argparse.ArgumentParser(description="Image processing tasks")

    # Common arguments
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--class', type=str, required=True, help='Class of the object to be segmented')
    parser.add_argument('--output', type=str, required=True, help='Path to the output image')

    # Task-specific arguments
    subparsers = parser.add_subparsers(dest='task', help='Task to perform')

    # Task 1: Generate red mask of the segmented object
    parser_task1 = subparsers.add_parser('task1', help='Generate red mask on the object')

    # Task 2: Change position of the segmented object
    parser_task2 = subparsers.add_parser('task2', help='Change position of the segmented object')
    parser_task2.add_argument('--x', type=int, required=True, help='Number of pixels to shift in the x direction')
    parser_task2.add_argument('--y', type=int, required=True, help='Number of pixels to shift in the y direction')

    # Task 2: Remove the object
    parser_task3 = subparsers.add_parser('task3', help='Remove the object')

    args = parser.parse_args()

    editor = ImageEditor(lama_config, lama_ckpt, mask_dilate_factor = 60)

    # Task dispatch
    if args.task == 'task1':
        # Call the function to handle Task 1
        editor.segment_anything(image_path = args.image, text_prompt = args.class, output_image_path = args.output)
    elif args.task == 'task2':
        # Call the function to handle Task 2
        editor.change_location_anything(image_path = args.image, text_prompt = args.class, shift_x = args.x, shift_y = args.y, output_image_path = args.output)
    elif args.task == 'task3':
        editor.segment_anything(image_path = args.image, text_prompt = args.class, output_image_path = args.output)
    else:
        print("Invalid task specified. Please choose 'task1' or 'task2'.")