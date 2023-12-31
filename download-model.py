import argparse
from huggingface_hub import list_repo_tree, snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download model from HuggingFace, default to safetensors only.")
    parser.add_argument('model', help='The model to download.')
    parser.add_argument('revision', nargs='?', default=None, help='The revision of the model to download.')
    parser.add_argument('-t', '--repo-type', choices=['model', 'dataset', 'space'], default=None,
                        help='The type of the repository from which to get the information.')
    parser.add_argument('-d', '--local-dir', default=None,
                        help='Local directory to store downloaded files.')
    parser.add_argument('-r', '--resume-download', action='store_true',
                        help='Resume a previously interrupted download.')
    parser.add_argument('-f', '--force-download', action='store_true',
                        help='Force download even if file already exists in cache.')
    parser.add_argument('-o', '--token', default=None,
                        help='Token for downloading files.')
    parser.add_argument('-a', '--allow-patterns', nargs='+', default=[],
                        help='Patterns to filter downloaded files.')
    parser.add_argument('-i', '--ignore-patterns', nargs='+', default=[],
                        help='Patterns to ignore during download.')
    parser.add_argument('-p', '--pt', action='store_true',
                        help='Download only PyTorch files(.pt, .ckpt, .bin).')
    parser.add_argument('-l', '--all', action='store_true',
                        help='Download all files.')
    parser.add_argument('-w', '--max-workers', default=69420, type=int,
                        help='Max workers to download files concurrently.')
    args = parser.parse_args()

    if args.pt and args.all:
        parser.error('`--pt-only` and `--all` are mutually exclusive.')

    if not args.all:
        print("Getting files from repo...")
        files_info = list_repo_tree(repo_id=args.model, revision=args.revision, recursive=True, repo_type=args.repo_type, token=args.token)
        has_safetensors = any(file_info.path.endswith('.safetensors') for file_info in files_info)
        if has_safetensors:
            if args.pt:
                args.ignore_patterns.append("*.safetensors")
                print("Downloading PyTorch files only...")
            else:
                args.ignore_patterns.append("*.pt")
                args.ignore_patterns.append("*.ckpt")
                args.ignore_patterns.append("*.bin")
                print("Downloading safetensors files only...")
        else:
            # args.ignore_patterns.append("*.safetensors")
            # if not args.pt:
            #     args.pt = True
            print("No safetensors file found in the repository, downloading PyTorch files only...")
    else:
        print("Downloading all files...")

    snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        repo_type=args.repo_type,
        local_dir=args.local_dir if args.local_dir else None,
        local_dir_use_symlinks=False if args.local_dir else 'auto',
        resume_download=args.resume_download,
        force_download=args.force_download,
        token=args.token,
        allow_patterns=args.allow_patterns if args.allow_patterns else None,
        ignore_patterns=args.ignore_patterns if args.ignore_patterns else None,
        max_workers=args.max_workers,
    )

if __name__ == "__main__":
    main()
