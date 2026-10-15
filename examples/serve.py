"""python examples/serve.py --checkpoint runs/low_resource/best.pt"""

import argparse

from guminet.inference.api import create_app

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args.checkpoint), host="127.0.0.1", port=args.port)
