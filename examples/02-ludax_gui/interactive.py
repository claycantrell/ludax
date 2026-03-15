import argparse
from ludax.gui import app
from ludax.policies.simple import random_policy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    app.config['POLICIES'] = {
        'random': random_policy(),
    }

    app.run(host=args.host, port=args.port, debug=True)
