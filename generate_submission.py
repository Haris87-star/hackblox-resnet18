"""Safe Kaggle submission entry point.

Delegates to predict.py so the 6,000-image undefined pool is never mistaken
for the 1,800-image test set.
"""

from predict import main


if __name__ == "__main__":
    raise SystemExit(main())
