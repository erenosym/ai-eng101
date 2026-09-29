import os
import tempfile

from markitdown import MarkItDown


converter = MarkItDown()


def parse_document(
    content: bytes,
    filename: str
) -> str:
    suffix = os.path.splitext(filename)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        result = converter.convert(temp_path)
        return result.markdown

    finally:
        os.remove(temp_path)