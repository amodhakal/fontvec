# fontvec

Take a folder of font files and generate a vector embedding for each file.

## Usage

```sh
# Recommended
uv run fontvec path/to/font

# For font images
uv run fontvec path/to/font --save-images path/to/image
```

## Fonts

The sample fonts in [`tests/fonts/`](tests/fonts/) come from [Google Fonts](https://fonts.google.com/).
They are included for testing and are **not** covered by this project's MIT
license.

## Acknowledgements

Inspired by [Fontjoy](https://github.com/Jack000/fontjoy) by Jack Qiao, which
explores font vectors and font pairing.

## License

MIT — see [LICENSE](LICENSE).
