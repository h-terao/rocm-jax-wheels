import argparse
from pathlib import Path
import subprocess
import tempfile
import re


def get_new_wheel_name(wheel_name: str, rocm_path: str) -> str:
    # e.g., `.dev20240713` from `jaxlib-0.4.30.dev20240713-cp312-cp312-manylinux2014_x86_64.whl`
    dev_version = re.match(r"jaxlib-\d+.\d+.\d+([^-]+)", wheel_name).group(1)
    # e.g., `613` from `/opt/rocm-6.1.3`
    rocm_version = "".join(re.match(r"[^-]+-(\d+).(\d+).(\d+)", rocm_path).groups())
    return wheel_name.replace(dev_version, "+rocm" + rocm_version)


def main(out_dir: str, jax_version: str, rocm_path: str, python_version: str) -> None:
    out_dir_path = Path(out_dir, f"v{jax_version}")
    if not out_dir_path.exists():
        out_dir_path.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as working_dir:
        working_dir_path = Path(working_dir)

        # build jaxlib with ROCm.
        build_command = (
            f"cd {working_dir} && "
            f"git clone https://github.com/ROCm/jax.git -b rocm-jaxlib-v{jax_version} --depth 1 && "
            "cd jax && "
            f"python build/build.py --enable_rocm --rocm_path={rocm_path} --python_version={python_version}"
        )
        subprocess.run([build_command], shell=True)

        # move the built wheel to the output directory.
        wheel_paths = list((working_dir_path / "jax" / "dist").glob("*.whl"))
        assert len(wheel_paths) == 1, f"Found wheels: {wheel_paths}"

        wheel_path = wheel_paths[0]
        new_wheel_name = get_new_wheel_name(wheel_path.name, rocm_path)
        wheel_path.rename(out_dir_path / new_wheel_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jax_version",
        "-jv",
        required=True,
        help="jaxlib version to be built.",
    )
    parser.add_argument(
        "--rocm_path",
        "-rocm",
        required=True,
        help="path to ROCm. It may starts from `/opt/`",
    )
    parser.add_argument(
        "--python_version",
        "-pv",
        default="3.12",
        help="Python version.",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        default="dist",
        help="Output directory to save the built wheel.",
    )

    args = parser.parse_args()
    main(
        out_dir=args.out_dir,
        jax_version=args.jax_version,
        rocm_path=args.rocm_path,
        python_version=args.python_version,
    )
