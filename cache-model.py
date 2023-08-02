import sys
from huggingface_hub import snapshot_download

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <model> [<revision>]")
        sys.exit(1)

    model = sys.argv[1]
    revision = None

    if len(sys.argv) > 2:
        revision = sys.argv[2]

    snapshot_download(repo_id=model, revision=revision)

if __name__ == '__main__':
    main()
