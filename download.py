from modelscope import snapshot_download
from modelscope.hub.file_download import model_file_download


def modelScopeDownloader(mode, model_name, file_name, output_fp, ignore=False):
    if mode == "single":
        model_file_download(
            model_id=model_name,
            file_path=file_name,
            cache_dir=output_fp,
    )
    elif mode == "whole":
        snapshot_download(
            model_id=model_name,
            ignore_file_pattern=[
                    #   '.pt',
                    #   ".bin",
                    #   '.safetensors',
                      ],
            cache_dir=output_fp,
            )

# MODEL_NAME = "LLM-Research/Meta-Llama-3-70B-Instruct"
# FILE_NAME = "model-00030-of-00030.safetensors"
# OUTPUT_FP = "/public/HYK/models"
# mode = "single"

# MODEL_NAME = "LLM-Research/Meta-Llama-3-70B-Instruct"
# FILE_NAME = ""
# OUTPUT_FP = "/public/HYK/models"
# mode = "whole"

MODEL_NAME = "Wojtek/flan-t5-large"
FILE_NAME = ""
OUTPUT_FP = "/public/HYK/models"
mode = "whole"


modelScopeDownloader(mode, MODEL_NAME, FILE_NAME, OUTPUT_FP, ignore=False)
