from _bootstrap import add_project_src_to_path

add_project_src_to_path()

from pathlib import Path

from mri_recon.data.multi_file_complex_dataset import (
    MultiFileFastMRIComplexSingleCoilDataset,
)


def main() -> None:
    data_dir = Path("data/knee_singlecoil_val")
    h5_paths = sorted(data_dir.glob("*.h5"))

    print(f"Found {len(h5_paths)} h5 files.")

    dataset = MultiFileFastMRIComplexSingleCoilDataset(
        h5_paths=h5_paths,
        acceleration=4,
        center_fraction=0.08,
        use_middle_slices_only=True,
        middle_slice_margin=5,
    )

    print(f"Total samples: {len(dataset)}")

    sample = dataset[0]

    print("Sample keys:")
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
