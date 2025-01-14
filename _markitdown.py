from typing import List, Union
import copy
import re
import traceback
from .document_converter_result import DocumentConverterResult
from .exceptions import FileConversionException, UnsupportedFormatException

class Markitdown:
    def _convert(
        self, local_path: str, extensions: List[Union[str, None]], **kwargs
    ) -> DocumentConverterResult:
        error_trace = ""
        for ext in extensions + [None]:  # Try last with no extension
            for converter in self._page_converters:
                _kwargs = copy.deepcopy(kwargs)

                # Overwrite file_extension appropriately
                if ext is None:
                    if "file_extension" in _kwargs:
                        del _kwargs["file_extension"]
                else:
                    _kwargs.update({"file_extension": ext})

                # Copy any additional global options
                if "mlm_client" not in _kwargs and self._mlm_client is not None:
                    _kwargs["mlm_client"] = self._mlm_client

                if "mlm_model" not in _kwargs and self._mlm_model is not None:
                    _kwargs["mlm_model"] = self._mlm_model

                # If we hit an error log it and keep trying
                try:
                    res = converter.convert(local_path, **_kwargs)
                    if res is not None:
                        # Normalize the content
                        res.text_content = "\n".join(
                            [line.rstrip() for line in re.split(r"\r?\n", res.text_content)]
                        )
                        res.text_content = re.sub(r"\n{3,}", "\n\n", res.text_content)
                        return res
                except Exception:
                    error_trace = ("\n\n" + traceback.format_exc()).strip()
                    continue

        # If we got this far without success, report any exceptions
        if len(error_trace) > 0:
            raise FileConversionException(
                f"Could not convert '{local_path}' to Markdown. File type was recognized as {extensions}. While converting the file, the following error was encountered:\n\n{error_trace}"
            )

        # Nothing can handle it!
        raise UnsupportedFormatException(
            f"Could not convert '{local_path}' to Markdown. The formats {extensions} are not supported."
        ) 