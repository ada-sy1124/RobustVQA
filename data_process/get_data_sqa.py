"""Backward-compatible entrypoint.

历史文件名 `get_data_sqa.py` 仍可用，实际逻辑已迁移到 `get_data_spa.py`。
"""

from get_data_spa import main


if __name__ == "__main__":
    main()
