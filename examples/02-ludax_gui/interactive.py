from ludax.gui import create_app


if __name__ == '__main__':
    app = create_app(games_folder="./games")
    app.run(port=8080, debug=True)