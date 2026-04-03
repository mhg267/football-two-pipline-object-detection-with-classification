from converter import FootballConverter


def main(mode):
    path = "/Users/minhhung/Documents/Code/Python/Computer Vision/Data/Dataset/Football"

    if mode not in ["train", "val", "test"]:
        raise ValueError("mode must be train/val/test")

    print(f"--- Converting {mode} ---")
    FootballConverter(path, mode)
    print(f"--- Done! ---")

if __name__ == '__main__':
    main(mode="train")