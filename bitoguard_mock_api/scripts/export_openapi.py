from __future__ import annotations

from pathlib import Path

import yaml

from app.main import create_app


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    app = create_app()
    schema = app.openapi()
    output_path = root / "openapi.yaml"
    output_path.write_text(yaml.safe_dump(schema, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
