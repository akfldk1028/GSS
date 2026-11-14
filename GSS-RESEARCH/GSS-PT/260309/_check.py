from pptx import Presentation
from pptx.util import Inches
import os

p = Presentation(r'C:\DK\GSS\GSS-RESEARCH\GSS-PT\260309\GSS_v3_korean.pptx')
print(f"Total slides: {len(p.slides)}")
print(f"Slide width: {p.slide_width}, height: {p.slide_height}")
print()

for i, slide in enumerate(p.slides):
    title = slide.shapes.title.text if slide.shapes.title else "(no title)"
    img_count = sum(1 for s in slide.shapes if s.shape_type == 13)  # 13 = picture
    print(f"Slide {i+1}: {title}")
    print(f"  Images: {img_count}")
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text and shape != slide.shapes.title:
            text_preview = shape.text[:80].replace('\n', ' | ')
            print(f"  Text: {text_preview}")
    print()
