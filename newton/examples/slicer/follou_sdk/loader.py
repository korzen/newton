import os, platform, pathlib, ctypes

NAMES = {
    "Windows":  "FollouDeviceController.dll",
    "Darwin":   "libFollouDeviceController.dylib",
    "Linux":    "libFollouDeviceController.so",
}

def load_lib(custom_path: str | None = None) -> ctypes.CDLL:
    """Locate the shared library and return a ctypes handle."""
    sys = platform.system()
    libname = NAMES.get(sys)
    if not libname:
        raise RuntimeError(f"Unsupported OS: {sys}")

    # explicit path ▸ env var ▸ repo root ▸ cwd
    search = []
    if custom_path:
        search.append(pathlib.Path(custom_path))
    if p := os.getenv("FOLLOU_LIB_PATH"):
        search.append(pathlib.Path(p))
    here = pathlib.Path(__file__).resolve().parent
    search.extend([
        here / libname,           # previously existing
        here / "dlls" / libname,  # <-- new: look in follou/dlls/
    ])
    # 4) repo root / cwd
    search.append(pathlib.Path.cwd() / libname)

    for p in search:
        if p.exists():
            return ctypes.CDLL(str(p))

    raise FileNotFoundError(f"Could not find {libname}.")