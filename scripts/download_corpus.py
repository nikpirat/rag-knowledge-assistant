"""One-time script to download AWS Well-Architected Framework whitepapers.

Not part of the package: a manual setup step, like Project 1's
download_data.py. Re-running is safe — skips files that already exist.
"""

import logging
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Confirmed working pattern (verified against reliability-pillar directly):
# https://docs.aws.amazon.com/pdfs/wellarchitected/latest/{slug}/wellarchitected-{slug}.pdf
PILLAR_SLUGS = [
    "operational-excellence-pillar",
    "security-pillar",
    "reliability-pillar",
    "performance-efficiency-pillar",
    "cost-optimization-pillar",
    "sustainability-pillar",
]
URL_TEMPLATE = (
    "https://docs.aws.amazon.com/pdfs/wellarchitected/latest/{slug}/wellarchitected-{slug}.pdf"
)
DEST = Path("data/raw")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for slug in PILLAR_SLUGS:
            target = DEST / f"{slug}.pdf"
            if target.exists():
                logger.info("Already have %s, skipping", target)
                continue

            url = URL_TEMPLATE.format(slug=slug)
            logger.info("Downloading %s", url)
            response = client.get(url)

            if response.status_code != 200:
                logger.error(
                    "Failed to download %s (status %d) — check the exact slug on "
                    "https://docs.aws.amazon.com/wellarchitected/latest/framework/welcome.html",
                    slug,
                    response.status_code,
                )
                continue

            target.write_bytes(response.content)
            logger.info("Saved %s (%d bytes)", target, len(response.content))


if __name__ == "__main__":
    main()
