"""
DermNet NZ Image Scraper

Scrapes dermatological condition images from dermnetnz.org for the
inflammatory skin conditions not available in the Kaggle DermNet dataset.

Images are saved to data/dermnet_scraped/<condition_name>/ with original
filenames preserved.
"""

import asyncio
import logging
import os
import time
from pathlib import Path

import httpx
from playwright.async_api import async_playwright

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Base URL
BASE = "https://dermnetnz.org"

# Gallery pages to scrape, grouped by output class label.
# Each class maps to one or more gallery URLs on DermNet NZ.
GALLERIES: dict[str, list[str]] = {
    "seborrheic_dermatitis": [
        f"{BASE}/images/seborrhoeic-dermatitis-images",
    ],
    "psoriasis": [
        f"{BASE}/images/chronic-plaque-psoriasis-images",
        f"{BASE}/topics/guttate-psoriasis-images",
        f"{BASE}/topics/nail-psoriasis-images",
        f"{BASE}/topics/palmoplantar-psoriasis-images",
        f"{BASE}/topics/genital-psoriasis-images",
    ],
    "atopic_dermatitis": [
        f"{BASE}/images/atopic-dermatitis-images",
        f"{BASE}/images/atopic-flexural-eczema-images",
    ],
    "acne": [
        f"{BASE}/images/acne-vulgaris-images",
    ],
    "basal_cell_carcinoma": [
        f"{BASE}/images/basal-cell-carcinoma-images",
    ],
    "melanoma": [
        f"{BASE}/images/melanoma-in-situ-images",
        f"{BASE}/images/nodular-melanoma-images",
        f"{BASE}/images/superficial-spreading-melanoma-images",
        f"{BASE}/images/acral-lentiginous-melanoma-images",
    ],
    "rosacea": [
        f"{BASE}/images/rosacea-images",
    ],
    "vitiligo": [
        f"{BASE}/images/vitiligo-images",
    ],
    "scabies": [
        f"{BASE}/images/scabies-images",
    ],
    "eczema": [
        f"{BASE}/images/discoid-eczema-images",
    ],
    "impetigo": [
        f"{BASE}/images/impetigo-images",
    ],
}

# Respectful delay between HTTP requests (seconds)
REQUEST_DELAY = 1.0

# Output root (relative to project root)
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "dataset" / "dermnet_nz"


async def extract_image_urls(page, url: str) -> list[str]:
    """Navigate to a gallery page and extract all clinical image URLs."""
    await page.goto(url, wait_until="networkidle")
    await page.wait_for_timeout(1000)

    urls = await page.eval_on_selector_all(
        "img[class*='js-gallery-image']",
        "imgs => imgs.map(img => img.src)",
    )
    log.info("Found %d images at %s", len(urls), url)
    return urls


async def download_images(
    image_urls: list[str],
    output_dir: Path,
    client: httpx.AsyncClient,
) -> int:
    """Download images to the output directory. Returns count of downloaded images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for url in image_urls:
        filename = url.split("/")[-1]
        filepath = output_dir / filename

        if filepath.exists():
            log.debug("Skipping (exists): %s", filename)
            downloaded += 1
            continue

        try:
            resp = await client.get(url)
            resp.raise_for_status()
            filepath.write_bytes(resp.content)
            downloaded += 1
            log.debug("Downloaded: %s", filename)
        except httpx.HTTPError as e:
            log.warning("Failed to download %s: %s", url, e)

        await asyncio.sleep(REQUEST_DELAY)

    return downloaded


async def scrape():
    """Main scraping entrypoint."""
    log.info("Output directory: %s", OUTPUT_DIR)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        # Accept cookies on first page load
        await page.goto(BASE, wait_until="networkidle")
        try:
            agree_btn = page.locator("button", has_text="AGREE")
            await agree_btn.click(timeout=5000)
            log.info("Accepted cookie consent")
        except Exception:
            log.info("No cookie consent dialog found")

        # Collect all image URLs per class
        all_urls: dict[str, list[str]] = {}
        for class_name, gallery_urls in GALLERIES.items():
            urls = []
            for gallery_url in gallery_urls:
                urls.extend(await extract_image_urls(page, gallery_url))
                await asyncio.sleep(REQUEST_DELAY)
            all_urls[class_name] = urls
            log.info("Class '%s': %d total images found", class_name, len(urls))

        await browser.close()

    # Download all images
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=30.0,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
    ) as client:
        total = 0
        for class_name, urls in all_urls.items():
            class_dir = OUTPUT_DIR / class_name
            count = await download_images(urls, class_dir, client)
            total += count
            log.info("Class '%s': %d/%d images downloaded", class_name, count, len(urls))

    log.info("Done. Total images downloaded: %d", total)

    # Print summary
    print("\n=== Scrape Summary ===")
    for class_name in GALLERIES:
        class_dir = OUTPUT_DIR / class_name
        n = len(list(class_dir.glob("*.jpg"))) if class_dir.exists() else 0
        print(f"  {class_name}: {n} images")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(scrape())
