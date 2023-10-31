import os
import torch
from tqdm import tqdm
import requests

#from memory_profiler import profile

# Decodes the image without passing through the upscaler. The resulting image will be the same size as the latent
# Thanks to Kevin Turner (https://github.com/keturn) we have a shortcut to look at the decoded image!
def make_linear_decode(model_version, device='cuda:0'):
    v1_4_rgb_latent_factors = [
        #   R       G       B
        [ 0.298,  0.207,  0.208],  # L1
        [ 0.187,  0.286,  0.173],  # L2
        [-0.158,  0.189,  0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
    ]

    if model_version[:5] == "sd-v1":
        rgb_latent_factors = torch.Tensor(v1_4_rgb_latent_factors).to(device)
    else:
        raise Exception(f"Model name {model_version} not recognized.")

    def linear_decode(latent):
        latent_image = latent.permute(0, 2, 3, 1) @ rgb_latent_factors
        latent_image = latent_image.permute(0, 3, 1, 2)
        return latent_image

    return linear_decode


def download_model(model_map,root):
    
    url = model_map[root.model_checkpoint]['url']

    # CLI dialogue to authenticate download
    if model_map[root.model_checkpoint]['requires_login']:
        print("This model requires an authentication token")
        print("Please ensure you have accepted the terms of service before continuing.")

        username = input("[What is your huggingface username?]: ")
        token = input("[What is your huggingface token?]: ")

        _, path = url.split("https://")

        url = f"https://{username}:{token}@{path}"

    # contact server for model
    print(f"..attempting to download {root.model_checkpoint}...this may take a while")
    ckpt_request = requests.get(url,stream=True)
    request_status = ckpt_request.status_code

    # inform user of errors
    if request_status == 403:
        raise ConnectionRefusedError("You have not accepted the license for this model.")
    elif request_status == 404:
        raise ConnectionError("Could not make contact with server")
    elif request_status != 200:
        raise ConnectionError(f"Some other error has ocurred - response code: {request_status}")

    # write to model path
    with open(os.path.join(root.models_path, root.model_checkpoint), 'wb') as model_file:
        file_size = int(ckpt_request.headers.get("Content-Length"))
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=root.model_checkpoint) as pbar:
            for chunk in ckpt_request.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    model_file.write(chunk)
                    pbar.update(len(chunk))


#@profile
def load_model(root, load_on_run_all=True, check_sha256=True, map_location="cuda"):

    import torch
    from ldm.util import instantiate_from_config
    from omegaconf import OmegaConf
    from transformers import logging
    logging.set_verbosity_error()

    try:
        ipy = get_ipython()
    except:
        ipy = 'could not get_ipython'

    if 'google.colab' in str(ipy):
        path_extend = "deforum-stable-diffusion"
    else:
        path_extend = ""

    model_map = {
        "Protogen_V2.2.ckpt": {
            'sha256': 'bb725eaf2ed90092e68b892a1d6262f538131a7ec6a736e50ae534be6b5bd7b1',
            'url': "https://huggingface.co/darkstorm2150/Protogen_v2.2_Official_Release/resolve/main/Protogen_V2.2.ckpt",
            'requires_login': False,
        },
        "Realistic_Vision_V5.1.safetensors": {
            'sha256': '00445494c80979e173c267644ea2d7c67a37fe3c50c9f4d5a161d8ecdd96cb2f',
            'url': 'https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1.safetensors',
            'requires_login': False,
            }
        }
    }

    # config path
    ckpt_config_path = root.custom_config_path if root.model_config == "custom" else os.path.join(root.configs_path, root.model_config)

    if os.path.exists(ckpt_config_path):
        pass
        #print(f"{ckpt_config_path} exists")
    else:
        #print(f"Warning: {ckpt_config_path} does not exist.")
        ckpt_config_path = os.path.join(path_extend,"configs",root.model_config)
        #print(f"Using {ckpt_config_path} instead.")
        
    ckpt_config_path = os.path.abspath(ckpt_config_path)

    # checkpoint path or download
    ckpt_path = root.custom_checkpoint_path if root.model_checkpoint == "custom" else os.path.join(root.models_path, root.model_checkpoint)
    ckpt_valid = True

    if os.path.exists(ckpt_path):
        pass
    elif 'url' in model_map[root.model_checkpoint]:
        download_model(model_map,root)
    else:
        print(f"Please download model checkpoint and place in {os.path.join(root.models_path, root.model_checkpoint)}")
        ckpt_valid = False
        
    print(f"config_path: {ckpt_config_path}")
    print(f"ckpt_path: {ckpt_path}")

    if check_sha256 and root.model_checkpoint != "custom" and ckpt_valid:
        try:
            import hashlib
            print("..checking sha256")
            with open(ckpt_path, "rb") as f:
                bytes = f.read() 
                hash = hashlib.sha256(bytes).hexdigest()
                del bytes
            if model_map[root.model_checkpoint]["sha256"] == hash:
                print("..hash is correct")
            else:
                print("..hash in not correct")
                print("..redownloading model")
                download_model(model_map,root)
        except:
            print("..could not verify model integrity")

    def load_model_from_config(config, ckpt, verbose=False, device='cuda', print_flag=False, map_location="cuda"):
        print(f"..loading model")
        _ , extension = os.path.splitext(ckpt)
        if extension.lower() == ".safetensors":
            import safetensors.torch
            pl_sd = safetensors.torch.load_file(ckpt, device=map_location)
        else:
            pl_sd = torch.load(ckpt, map_location=map_location)
        try:
            sd = pl_sd["state_dict"]
        except:
            sd = pl_sd
        torch.set_default_dtype(torch.float16)
        model = instantiate_from_config(config.model)
        torch.set_default_dtype(torch.float32)
        m, u = model.load_state_dict(sd, strict=False)
        if print_flag:
            if len(m) > 0 and verbose:
                print("missing keys:")
                print(m)
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u)

        model = model.half().to(device)
        model.eval()
        return model

    if load_on_run_all and ckpt_valid:
        local_config = OmegaConf.load(f"{ckpt_config_path}")
        model = load_model_from_config(local_config, f"{ckpt_path}", map_location)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

    autoencoder_version = "sd-v1" #TODO this will be different for different models
    model.linear_decode = make_linear_decode(autoencoder_version, device)

    return model, device


def get_model_output_paths(root):

    models_path = root.models_path
    output_path = root.output_path

    #@markdown **Google Drive Path Variables (Optional)**
    
    force_remount = False

    try:
        ipy = get_ipython()
    except:
        ipy = 'could not get_ipython'

    if 'google.colab' in str(ipy):
        if root.mount_google_drive:
            from google.colab import drive # type: ignore
            try:
                drive_path = "/content/drive"
                drive.mount(drive_path,force_remount=force_remount)
                models_path = root.models_path_gdrive
                output_path = root.output_path_gdrive
            except:
                print("..error mounting drive or with drive path variables")
                print("..reverting to default path variables")

    models_path = os.path.abspath(models_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    print(f"models_path: {models_path}")
    print(f"output_path: {output_path}")

    return models_path, output_path
