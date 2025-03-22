# patchs_unsloth_llamacpp

## Overview

patchs_unsloth_llamacpp is a tool designed to accelerate model inference on the Windows platform using llama.cpp as its backend.

## Features

- Accelerates model inference by llama.cpp.
- Compatible with Windows operating system.
- If RAM is not enough, also support move data to VRAM ( this may consume more VRAM ).
- Only Support GRPOTrainer now.
- Automatically download latest llama.cpp from github to ./Llama.

## Usage Example

To use patchs_unsloth_llamacpp, follow these steps:

1. Install the python package:

    ```bash
    python3 -m pip install patchs_unsloth_llamacpp
    ```

2. Add code below in you project:

   ```python
   # use code below before import unsloth and unsloth-zoo
   from patchs_unsloth_llamacpp import patch
   patch("GRPO")
   ```

3. Run your project:

   ```bash
   python3 your_project.py
   ```

4. If RAM is OOM when training, add these code below to move data to VRAM instead.

    ```python
    from patchs_unsloth_llamacpp import unsloth_cpu_oom_patch
    unsloth_cpu_oom_patch(True)
    ```

## QAs

### 1. **First Run Error**

On the initial run, you might encounter an error stating "AttributeError: 'NoneType' object has no attribute 'span'". Simply running the application again should resolve this issue.

### 2. **GGUF Error**

As llama.cpp has it own gguf format to save models, this package has its own gguf python package, which may conflicts with other gguf packages installed in the system. If this happens, please uninstall or update guff package installed.

### 3. **Install llama.cpp Error**

If something wrong with installing llama.cpp to your project, please shutdown windows defender first ( Windows Defender may determine file risk ) and check the Internet connecting to Github. Or you can Install llama.cpp in Github ( https://github.com/ggml-org/llama.cpp/releases/latest )

## Contact Author

For questions or contributions, please contact the author at [liuzhi1999@foxmail.com](mailto:liuzhi1999@foxmail.com).
