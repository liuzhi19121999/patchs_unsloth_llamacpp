from pathlib import Path

def is_use_cuda_memory(state = False):
    from .change_file.unsloth_zoo_content import unsloth_zoo_content
    if state:
        unsloth_zoo_content = unsloth_zoo_content.replace('saved_hidden_states = hidden_states.to("cpu", non_blocking = True)', 'saved_hidden_states = hidden_states.to("cuda:0", non_blocking = True)')
    return unsloth_zoo_content

def write_file_content(file_path: str, content: str):
    with open(file_path, 'w', encoding='utf-8') as fileObj:
        fileObj.write(content)

def download_latest_llama():
    from requests import get
    from json import loads
    from zipfile import ZipFile
    from re import compile
    from torch import __version__
    from os import remove
    cuda_version = __version__.split('+')[-1]
    if cuda_version not in ["cu118", "cu124", "cu126"]:
        raise ValueError("No suitable cuda version found, eg. cu118 cu124 cu126 in torch.__version__")
    resp = get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
    downloads = loads(resp.content)["assets"]
    for download in downloads:
        package_name = download["name"].replace('.', '')
        download_link = download["browser_download_url"]
        cuda_version_re = compile(r'cu[0-9]+')
        cudas = cuda_version_re.findall(package_name)
        if "cudart" in package_name:
            continue
        if cudas.__len__() != 1 or "win" not in package_name:
            continue
        if cudas[0] <= cuda_version:
            if cuda_version != "cu118" and cudas[0] < "cu120":
                continue
            print("Downloading llama.cpp from  [ " + download_link + " ]")
            download_file = get(download_link, stream=True, allow_redirects=True)
            with open("./llamacpp.zip", 'wb') as f:
                for chunk in download_file.iter_content(2048):
                    f.write(chunk)
            print("Installing llama.cpp to ./Llama")
            ZipFile("./llamacpp.zip", 'r').extractall("./Llama")
            remove("./llamacpp.zip")
            return

def patch(typename: str = "GRPO"):
    """
    modify the trl and transformers to enable llama.cpp acceleration on windows.\n
    typename : the trainer type need to be changed
    """
    from os.path import exists
    from os import mkdir
    from transformers import trainer
    from trl.trainer import grpo_trainer
    from .change_file.trainer_content import Trainer_content
    if not exists("./Llama"):
        mkdir("./Llama")
    if not exists("./Llama/models"):
        mkdir("./Llama/models")
    if not exists("./Llama/llama-server.exe"):
        try:
            download_latest_llama()
        except Exception:
            print("Error : “./Llama/llama-server.exe” Not Found\nAs llama.cpp is an open source project, please download suitable llama.cpp version from https://github.com/ggml-org/llama.cpp/releases into directory ./Llama")
            exit(0)
    write_file_content(Path(trainer.__file__).absolute().__str__(), Trainer_content)
    if typename == "GRPO":
        from .change_file.grpo_content import GRPO_content
        write_file_content(Path(grpo_trainer.__file__).absolute().__str__(), GRPO_content)
    print(f"Patch {typename} trainer success! llama.cpp backend is enable on windows")

def unsloth_cpu_oom_patch(is_cpu_oom: bool = False):
    '''
    move data from RAM to VRAM, if you have less RAM than VRAM.\n
    is_cpu_oom : if RAM is OOM, this choice can move data to GPU
    '''
    import unsloth
    from unsloth_zoo import gradient_checkpointing
    unsloth_zoo_content = is_use_cuda_memory(is_cpu_oom)
    write_file_content(Path(gradient_checkpointing.__file__).absolute().__str__(), unsloth_zoo_content)
    print("Patch unsloth RAM reduce success! You can save training data in VRAM instead of RAM")
