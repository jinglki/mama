import mama.config

def download_from_huggingface(cfg, parser):
    from huggingface_hub import snapshot_download
    repo_id = cfg["model"].get("repo_id")
    if not repo_id:
        parser.error("model.repo_id is required for download")
    destination = mama.config.project_path(cfg["model"]["path"])
    snapshot_download(repo_id=repo_id, local_dir=destination)
    print(f"Model downloaded: {destination}")
    return

def download_from_modelscope(cfg, parser):
    import modelscope as ms
    repo_id = cfg["model"].get("repo_id")
    if not repo_id:
        parser.error("model.repo_id is required for download")
    destination = mama.config.project_path(cfg["model"]["path"])
    ms.snapshot_download(repo_id=repo_id, local_dir=destination.__str__())
    print(f"Model downloaded: {destination}")
    return