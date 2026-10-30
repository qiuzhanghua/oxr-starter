"""oxr-starter：在本机用 OXR-1.0 把 PDF 转成 Markdown / JSON。"""

from .api import Document, ImageRef, OXRClient, Page, Region, parse

__all__ = ["parse", "OXRClient", "Document", "Page", "Region", "ImageRef"]
