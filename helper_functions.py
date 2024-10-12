from pathlib import Path


def save_uploaded_file(
    uploaded_file: "UploadedFile", output_dir: Path = Path("/tmp")
) -> Path:
    output_path = Path(output_dir) / uploaded_file.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return output_path