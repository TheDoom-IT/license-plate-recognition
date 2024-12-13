import os

import argparse

def main(src_dir, src_type, prefix = None):
    prefix = prefix if prefix else src_dir.split("\\")[-1]
    images = []
    for country in os.listdir(src_dir):
        if not os.path.isdir(os.path.join(src_dir, country)):
            continue
        
        images += list(
            map(
                lambda x: "{}/{}/{}/{}".format(prefix, country, src_type, x), 
                filter(lambda x: x.endswith(".jpg"), os.listdir(os.path.join(src_dir, country, src_type)))
            )
        )
        
    with open(os.path.join(src_dir, f"{src_type}.txt"), "w") as f:
        f.write("\n".join(images))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize image to 416x416.")
    parser.add_argument("src_dir", type=str, help="Directory of dataset.")
    parser.add_argument("src_type", type=str, help="Type of directory (train, valid, or test).")
    parser.add_argument("prefix", type=str, default=None, help="Prefix for images.")

    args = parser.parse_args()
    main(args.src_dir, args.src_type, args.prefix)