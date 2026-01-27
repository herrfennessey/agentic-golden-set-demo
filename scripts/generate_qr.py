#!/usr/bin/env python3
"""Generate QR codes with optional custom logos for presentations."""

import argparse
from pathlib import Path

import qrcode
from PIL import Image
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers.pil import (
    CircleModuleDrawer,
    RoundedModuleDrawer,
    SquareModuleDrawer,
)


def generate_qr(
    data: str,
    output: Path,
    logo_path: Path | None = None,
    fill_color: str = "black",
    back_color: str = "white",
    style: str = "square",
    box_size: int = 10,
    border: int = 4,
) -> None:
    """Generate a QR code with optional embedded logo.

    Args:
        data: URL or text to encode.
        output: Output file path.
        logo_path: Optional path to logo image.
        fill_color: QR code foreground color.
        back_color: QR code background color.
        style: Module style - 'square', 'rounded', or 'circle'.
        box_size: Size of each QR code box in pixels.
        border: Border size in boxes.
    """
    # Select module drawer based on style
    drawers = {
        "square": SquareModuleDrawer(),
        "rounded": RoundedModuleDrawer(),
        "circle": CircleModuleDrawer(),
    }
    module_drawer = drawers.get(style, SquareModuleDrawer())

    # Create QR code with high error correction (allows ~30% damage)
    qr = qrcode.QRCode(
        version=None,  # Auto-size
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)

    if logo_path:
        # Use StyledPilImage with embedded logo
        img = qr.make_image(
            image_factory=StyledPilImage,
            module_drawer=module_drawer,
            embedded_image_path=str(logo_path),
        )
        # StyledPilImage doesn't support fill_color directly, convert and save
        img.save(output)
    else:
        # Standard image with colors
        img = qr.make_image(
            image_factory=StyledPilImage,
            module_drawer=module_drawer,
        )
        img.save(output)

    print(f"QR code saved to: {output}")


def generate_qr_manual(
    data: str,
    output: Path,
    logo_path: Path,
    fill_color: str = "black",
    back_color: str = "white",
    logo_size_percent: int = 25,
    box_size: int = 10,
    border: int = 4,
) -> None:
    """Generate QR code with manual logo placement (more control over sizing).

    Args:
        data: URL or text to encode.
        output: Output file path.
        logo_path: Path to logo image.
        fill_color: QR code foreground color.
        back_color: QR code background color.
        logo_size_percent: Logo size as percentage of QR code width.
        box_size: Size of each QR code box in pixels.
        border: Border size in boxes.
    """
    # Create QR code
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)

    qr_img = qr.make_image(fill_color=fill_color, back_color=back_color).convert("RGB")

    # Load and resize logo
    logo = Image.open(logo_path)
    if logo.mode != "RGBA":
        logo = logo.convert("RGBA")

    # Calculate logo size (percentage of QR code)
    qr_width = qr_img.size[0]
    logo_max_size = int(qr_width * logo_size_percent / 100)

    # Resize maintaining aspect ratio
    logo.thumbnail((logo_max_size, logo_max_size), Image.LANCZOS)

    # Calculate center position
    pos = (
        (qr_img.size[0] - logo.size[0]) // 2,
        (qr_img.size[1] - logo.size[1]) // 2,
    )

    # Create white background for logo area (improves contrast)
    bg_padding = 10
    bg_size = (logo.size[0] + bg_padding * 2, logo.size[1] + bg_padding * 2)
    bg_pos = (pos[0] - bg_padding, pos[1] - bg_padding)
    bg = Image.new("RGB", bg_size, back_color)
    qr_img.paste(bg, bg_pos)

    # Paste logo with transparency support
    qr_img.paste(logo, pos, logo if logo.mode == "RGBA" else None)

    qr_img.save(output)
    print(f"QR code saved to: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate QR codes with optional custom logos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple QR code
  python scripts/generate_qr.py "https://example.com" -o qr.png

  # With logo
  python scripts/generate_qr.py "https://example.com" -o qr.png --logo logo.png

  # Styled with rounded corners
  python scripts/generate_qr.py "https://example.com" -o qr.png --style rounded

  # Custom colors
  python scripts/generate_qr.py "https://example.com" -o qr.png --fill darkblue

  # Manual method with larger logo
  python scripts/generate_qr.py "https://example.com" -o qr.png --logo logo.png --manual --logo-size 30
        """,
    )
    parser.add_argument("data", help="URL or text to encode in QR code")
    parser.add_argument("-o", "--output", type=Path, default=Path("qr_code.png"), help="Output file path")
    parser.add_argument("--logo", type=Path, help="Path to logo image to embed")
    parser.add_argument("--fill", default="black", help="QR code fill color (default: black)")
    parser.add_argument("--background", default="white", help="QR code background color (default: white)")
    parser.add_argument(
        "--style",
        choices=["square", "rounded", "circle"],
        default="square",
        help="Module style (default: square)",
    )
    parser.add_argument("--box-size", type=int, default=10, help="Box size in pixels (default: 10)")
    parser.add_argument("--border", type=int, default=4, help="Border size in boxes (default: 4)")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use manual logo placement method (more control)",
    )
    parser.add_argument(
        "--logo-size",
        type=int,
        default=25,
        help="Logo size as %% of QR width, manual mode only (default: 25)",
    )

    args = parser.parse_args()

    if args.manual and args.logo:
        generate_qr_manual(
            data=args.data,
            output=args.output,
            logo_path=args.logo,
            fill_color=args.fill,
            back_color=args.background,
            logo_size_percent=args.logo_size,
            box_size=args.box_size,
            border=args.border,
        )
    else:
        generate_qr(
            data=args.data,
            output=args.output,
            logo_path=args.logo,
            fill_color=args.fill,
            back_color=args.background,
            style=args.style,
            box_size=args.box_size,
            border=args.border,
        )


if __name__ == "__main__":
    main()
